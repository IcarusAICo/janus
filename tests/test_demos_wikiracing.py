"""Offline wikiracing parser and two-stage shortlist tests. No Wikipedia, no models."""
from demos.wikiracing.race import choose_next, shortlist_from_scores
from demos.wikiracing.wikipedia import extract_article_links

BASEBALL = """
<html><body>
<div id="mw-content-text">
<p>See also <a href="/wiki/Scientific_American">Scientific American</a> and
<a href="https://en.wikipedia.org/wiki/Sun">Sun</a>.</p>
<p>Skip <a href="/wiki/Baseball">Baseball</a>,
<a href="/wiki/Help:Contents">help</a>,
<a href="https://example.com">external</a>,
<a href="/wiki/Foo#section">Foo</a>.</p>
</div>
</body></html>
"""


def test_extract_article_links_keeps_enwiki_article_titles_in_order():
    links = extract_article_links(BASEBALL, current_title="Baseball")
    assert [l.title for l in links] == ["Scientific American", "Sun"]
    assert all(l.href.startswith("/wiki/") for l in links)


def test_direct_target_link_is_taken_without_a_model_call():
    links = extract_article_links(BASEBALL, current_title="Baseball")
    picked, stage = choose_next(links, target_title="Sun", scores=None, choice=None)
    assert picked.title == "Sun" and stage == "direct"


def test_choice_under_255_uses_the_model_choice_key():
    links = extract_article_links(BASEBALL, current_title="Baseball")
    picked, stage = choose_next(links, target_title="Moon", scores=None, choice="Scientific_American")
    assert picked.title == "Scientific American" and stage == "choice"


def test_high_cardinality_scores_then_shortlists():
    titles = [f"Link {i}" for i in range(300)]
    scores = {f"l{i}": (4.0 if i == 7 else 0.2) for i in range(300)}
    top = shortlist_from_scores(titles, scores, k=32)
    assert top[0] == "Link 7"
    assert len(top) == 32
