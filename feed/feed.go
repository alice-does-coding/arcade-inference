// Package feed fetches headlines from curated sources for the simulation.
// Diet matters: agents react in-character to what they read, so we feed them
// things that make a vending machine or a lantern actually interesting.
//
// Sources: Reddit (when it cooperates) + Hacker News (always works).
// Circuit breaker: a source that fails 5 times in a row is skipped for 3
// refresh cycles (~45 min) before being retried.
package feed

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand/v2"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"
)

const RefreshInterval = 15 * time.Minute
const circuitBreakAfter = 5
const circuitCooldown = 3 * RefreshInterval // 45 min before retry

type Headline struct {
	Title    string `json:"title"`
	Summary  string `json:"summary"`
	Category string `json:"category"`
	Source   string `json:"source"`
	URL      string `json:"url"`
}

// source represents a fetchable news source.
type source struct {
	name     string
	category string
	fetch    func() ([]Headline, error)
}

// redditSubs is the curated list of subreddits.
// Selected for text-rich personal-drama posts that things can have actual
// reactions to. The previous list (philosophy, neuroscience, MachineLearning,
// etc.) was too academic — things are not rocket scientists; they're things.
// Drama, advice, confessions, casual life stuff fuel personality-driven
// reactions way better than tidy ideas.
var redditSubs = []struct{ name, category string }{
	// The drama core — text-rich, opinion-bait, things have actual takes
	{"AmItheAsshole", "Drama"},
	{"relationship_advice", "Advice"},
	{"tifu", "Drama"},
	{"confession", "Drama"},
	{"maliciouscompliance", "Drama"},
	{"pettyrevenge", "Drama"},

	// Curiosity ("huh, look at that") — kept because the title alone often
	// carries enough context for a thing to react. Filtered to text-only
	// posts, so image/video posts from these subs don't make it through.
	{"interestingasfuck", "Curiosity"},
	{"mildlyinteresting", "Curiosity"},
}

// redditBlocklist filters out titles touching genuinely heavy content.
// Tightened 2026-04-26: previous list also blocked "abuse", "assault",
// "harassment", "depression", "alcohol", etc. — bread-and-butter vocabulary
// of AITA / relationship_advice. With the drama subs added, blocking those
// would filter out half the feed. Hard blocks only now (sexual violence,
// child harm, mass violence, self-harm, hard-drug death, politics).
var redditBlocklist = []string{
	// Self-harm + suicide
	"suicide", "suicidal", "kill myself", "killed myself", "took my life", "took her life", "took his life",
	// Sexual violence
	"rape", "raped", "sexual assault", "molest", "molested", "groomed", "grooming",
	// Child harm
	"child abuse", "child porn", "csam",
	// Mass / weapon violence
	"mass shooting", "school shooting", "terrorist", "terrorism", "bombing", "missile attack",
	// Hard-drug death
	"fentanyl", "overdosed", "heroin overdose",
	// Politics — drama subs do mention these but the threads turn nasty fast
	"trump", "biden", "harris", "vance",
	"congress", "senate", "republican", "democrat",
	"election", "ballot", "politician",
	// Scams / predatory finance
	"scam", "scammer",
}

var blockRe *regexp.Regexp

func init() {
	escaped := make([]string, len(redditBlocklist))
	for i, w := range redditBlocklist {
		escaped[i] = regexp.QuoteMeta(w)
	}
	blockRe = regexp.MustCompile(`(?i)\b(` + strings.Join(escaped, "|") + `)\b`)
}

func isTitleOK(title string) bool {
	return !blockRe.MatchString(title)
}

var client = &http.Client{Timeout: 8 * time.Second}

// Cache holds fetched headlines, refreshes periodically, and tracks per-source failures.
type Cache struct {
	mu        sync.RWMutex
	fetchMu   sync.Mutex
	headlines []Headline
	fetchedAt time.Time

	// Circuit breaker state — protected by fetchMu.
	failures  map[string]int
	deadUntil map[string]time.Time
}

var Default = &Cache{
	failures:  make(map[string]int),
	deadUntil: make(map[string]time.Time),
}

// Sample returns n random headlines, refreshing the cache if stale.
func (c *Cache) Sample(n int) []Headline {
	c.refreshIfStale()
	c.mu.RLock()
	defer c.mu.RUnlock()
	if len(c.headlines) == 0 {
		return nil
	}
	pool := make([]Headline, len(c.headlines))
	copy(pool, c.headlines)
	rand.Shuffle(len(pool), func(i, j int) { pool[i], pool[j] = pool[j], pool[i] })
	if n > len(pool) {
		n = len(pool)
	}
	return pool[:n]
}

func (c *Cache) refreshIfStale() {
	c.mu.RLock()
	stale := time.Since(c.fetchedAt) >= RefreshInterval || len(c.headlines) == 0
	c.mu.RUnlock()
	if !stale {
		return
	}

	c.fetchMu.Lock()
	defer c.fetchMu.Unlock()
	c.mu.RLock()
	stale = time.Since(c.fetchedAt) >= RefreshInterval || len(c.headlines) == 0
	c.mu.RUnlock()
	if !stale {
		return
	}

	var fresh []Headline
	now := time.Now()

	// Reddit sources.
	for _, sub := range redditSubs {
		key := "r/" + sub.name
		if until, dead := c.deadUntil[key]; dead && now.Before(until) {
			continue // circuit open — skip
		}
		items, err := fetchReddit(sub.name, sub.category)
		if err != nil {
			c.failures[key]++
			if c.failures[key] >= circuitBreakAfter {
				c.deadUntil[key] = now.Add(circuitCooldown)
				c.failures[key] = 0
				log.Printf("[news] %s: circuit open after %d failures — pausing for %.0fm", key, circuitBreakAfter, circuitCooldown.Minutes())
			} else {
				log.Printf("[news] %s: %v (failure %d/%d)", key, err, c.failures[key], circuitBreakAfter)
			}
			continue
		}
		c.failures[key] = 0 // reset on success
		fresh = append(fresh, items...)
	}

	// Hacker News — always attempted, circuit breaker applies.
	hnKey := "hn/top"
	if until, dead := c.deadUntil[hnKey]; !dead || now.After(until) {
		items, err := fetchHN()
		if err != nil {
			c.failures[hnKey]++
			if c.failures[hnKey] >= circuitBreakAfter {
				c.deadUntil[hnKey] = now.Add(circuitCooldown)
				c.failures[hnKey] = 0
				log.Printf("[news] hn/top: circuit open after %d failures", circuitBreakAfter)
			} else {
				log.Printf("[news] hn/top: %v (failure %d/%d)", err, c.failures[hnKey], circuitBreakAfter)
			}
		} else {
			c.failures[hnKey] = 0
			fresh = append(fresh, items...)
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if len(fresh) > 0 {
		rand.Shuffle(len(fresh), func(i, j int) { fresh[i], fresh[j] = fresh[j], fresh[i] })
		c.headlines = fresh
		c.fetchedAt = now
		log.Printf("[news] cache refreshed — %d headlines", len(fresh))
	} else {
		// Keep stale headlines rather than going dark.
		log.Printf("[news] all sources failed — keeping %d stale headlines", len(c.headlines))
		c.fetchedAt = now // prevent tight retry loop
	}
}

// fetchReddit pulls hot text-only posts from a subreddit via Reddit's
// public JSON endpoint (no auth required).
//
// Filters to is_self == true (self/text posts only) — link posts, image
// posts, video posts, gallery posts all get skipped. Things have nothing
// to react to from a bare image URL with no caption.
func fetchReddit(subreddit, category string) ([]Headline, error) {
	url := fmt.Sprintf("https://www.reddit.com/r/%s/hot.json?limit=50&raw_json=1", subreddit)
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "garden-arcade/1.0 (simulation feed reader; contact alice@gardenarcade.ai)")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 2*1024*1024))
	if err != nil {
		return nil, err
	}

	var listing struct {
		Data struct {
			Children []struct {
				Data struct {
					Title     string `json:"title"`
					Selftext  string `json:"selftext"`
					IsSelf    bool   `json:"is_self"`
					IsVideo   bool   `json:"is_video"`
					Permalink string `json:"permalink"`
					Stickied  bool   `json:"stickied"`
				} `json:"data"`
			} `json:"children"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &listing); err != nil {
		return nil, fmt.Errorf("json parse: %w", err)
	}

	var out []Headline
	for _, c := range listing.Data.Children {
		p := c.Data
		// Text-only posts. Skip link / image / video / gallery / stickied
		// announcements. Some "self" posts have empty selftext (just a
		// title) — those are still allowed because the title itself often
		// carries the whole story (especially in AITA/tifu titles).
		if !p.IsSelf || p.IsVideo || p.Stickied {
			continue
		}
		title := strings.TrimSpace(p.Title)
		if title == "" || !isTitleOK(title) {
			continue
		}
		// Block on selftext too — heavy stuff sometimes hides behind a
		// neutral title.
		if p.Selftext != "" && !isTitleOK(p.Selftext) {
			continue
		}
		postURL := "https://www.reddit.com" + p.Permalink
		out = append(out, Headline{
			Title:    title,
			Summary:  extractTextSnippet(p.Selftext, 180),
			Category: category,
			Source:   "r/" + subreddit,
			URL:      postURL,
		})
	}
	return out, nil
}

var htmlTagRe = regexp.MustCompile(`<[^>]+>`)
var multiSpaceRe = regexp.MustCompile(`\s{2,}`)

// extractTextSnippet strips HTML tags from s and returns up to maxLen characters.
func extractTextSnippet(s string, maxLen int) string {
	s = htmlTagRe.ReplaceAllString(s, " ")
	s = multiSpaceRe.ReplaceAllString(s, " ")
	s = strings.TrimSpace(s)
	// Remove the Reddit boilerplate "submitted by /u/..." lines
	if idx := strings.Index(s, "submitted by"); idx > 0 {
		s = strings.TrimSpace(s[:idx])
	}
	if len([]rune(s)) > maxLen {
		r := []rune(s)
		s = string(r[:maxLen]) + "…"
	}
	return s
}

// fetchHN pulls the current top stories from Hacker News via Firebase API.
// No auth required. Category is inferred from story domain.
func fetchHN() ([]Headline, error) {
	const maxStories = 40
	const fetchTop = 100

	req, err := http.NewRequest(http.MethodGet, "https://hacker-news.firebaseio.com/v0/topstories.json", nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("topstories: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("topstories: HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	if err != nil {
		return nil, err
	}
	var ids []int
	if err := json.Unmarshal(body, &ids); err != nil {
		return nil, err
	}
	if len(ids) > fetchTop {
		ids = ids[:fetchTop]
	}

	var mu sync.Mutex
	var wg sync.WaitGroup
	var out []Headline

	sem := make(chan struct{}, 10) // 10 concurrent fetches
	for _, id := range ids {
		wg.Add(1)
		go func(storyID int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			h, err := fetchHNStory(storyID)
			if err != nil || h == nil {
				return
			}
			mu.Lock()
			out = append(out, *h)
			mu.Unlock()
		}(id)
	}
	wg.Wait()

	if len(out) > maxStories {
		rand.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
		out = out[:maxStories]
	}
	return out, nil
}

func fetchHNStory(id int) (*Headline, error) {
	url := fmt.Sprintf("https://hacker-news.firebaseio.com/v0/item/%d.json", id)
	resp, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var story struct {
		Title string `json:"title"`
		URL   string `json:"url"`
		Score int    `json:"score"`
		Type  string `json:"type"`
		Dead  bool   `json:"dead"`
		Deleted bool `json:"deleted"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&story); err != nil {
		return nil, err
	}
	if story.Dead || story.Deleted || story.Type != "story" || story.Score < 50 {
		return nil, nil
	}
	title := strings.TrimSpace(story.Title)
	if title == "" || !isTitleOK(title) {
		return nil, nil
	}
	storyURL := story.URL
	if storyURL == "" {
		storyURL = fmt.Sprintf("https://news.ycombinator.com/item?id=%d", id)
	}
	return &Headline{
		Title:    title,
		Category: hnCategory(storyURL),
		Source:   "HN",
		URL:      storyURL,
	}, nil
}

// hnCategory infers a rough category from the story URL domain.
func hnCategory(url string) string {
	url = strings.ToLower(url)
	switch {
	case strings.Contains(url, "github") || strings.Contains(url, "gitlab"):
		return "Technology"
	case strings.Contains(url, "arxiv") || strings.Contains(url, "nature.com") || strings.Contains(url, "science"):
		return "Science"
	case strings.Contains(url, "nasa") || strings.Contains(url, "space"):
		return "Space"
	case strings.Contains(url, "music") || strings.Contains(url, "audio") || strings.Contains(url, "sound"):
		return "Music"
	case strings.Contains(url, "art") || strings.Contains(url, "design") || strings.Contains(url, "creative"):
		return "Art"
	default:
		return "Ideas"
	}
}
