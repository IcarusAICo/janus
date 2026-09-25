"""English Wikipedia article-link extraction. No search API."""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import unquote, urljoin, urlparse
import time
import urllib.request

WIKI_ORIGIN = "https://en.wikipedia.org"
SKIP_PREFIXES = (
    "Help:", "File:", "Image:", "Wikipedia:", "Special:", "Talk:", "Template:",
    "Template_talk:", "Category:", "Portal:", "Draft:", "User:", "MediaWiki:",
    "Module:", "TimedText:", "Book:",
)


@dataclass(frozen=True)
class WikiLink:
    title: str
    href: str
    key: str


def title_key(title):
    return title.replace(" ", "_")


def title_from_href(href):
    parsed = urlparse(href)
    if parsed.fragment or parsed.query:
        return None
    if parsed.netloc and parsed.netloc not in {"en.wikipedia.org", "www.wikipedia.org"}:
        return None
    path = unquote(parsed.path)
    if path.startswith("./"):
        path = "/wiki/" + path[2:]
    if not path.startswith("/wiki/"):
        return None
    slug = path[len("/wiki/"):]
    if not slug or ":" in slug:
        return None
    return slug.replace("_", " ")


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_content = False
        self.depth = 0
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "div" and attrs.get("id") == "mw-content-text":
            self.in_content = True
            self.depth = 1
            return
        if self.in_content and tag == "div":
            self.depth += 1
        if self.in_content and tag == "a":
            href = attrs.get("href")
            if href:
                self.hrefs.append(href)

    def handle_endtag(self, tag):
        if self.in_content and tag == "div":
            self.depth -= 1
            if self.depth <= 0:
                self.in_content = False


def extract_article_links(html, current_title):
    parser = _LinkParser()
    parser.feed(html)
    seen = set()
    links = []
    current = current_title.replace("_", " ").lower()
    for href in parser.hrefs:
        if urlparse(href).fragment or urlparse(href).query:
            continue
        title = title_from_href(href)
        if not title or title.lower() == current:
            continue
        key = title_key(title)
        if key in seen:
            continue
        seen.add(key)
        path = urlparse(href).path
        links.append(WikiLink(title=title, href=path if path.startswith("/wiki/") else href, key=key))
    return links


def fetch_article(title, opener=None, pause=0.15):
    """GET the live article HTML. Caller should pause between pages."""
    slug = title_key(title)
    url = urljoin(WIKI_ORIGIN, f"/wiki/{slug}")
    time.sleep(pause)
    req = urllib.request.Request(url, headers={"User-Agent": "jev-demos/0.1 (wikiracing; research)"})
    handle = opener.open(req) if opener else urllib.request.urlopen(req, timeout=30)
    with handle as incoming:
        final = incoming.geturl()
        html = incoming.read().decode("utf-8", errors="replace")
    canonical = unquote(urlparse(final).path.rsplit("/", 1)[-1]).replace("_", " ")
    return canonical, html
