"""The page at /: required elements, offline-safe, no unsafe HTML injection."""
import re

import pytest


@pytest.fixture(scope="module")
def html(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    return r.text


def test_form_fields_and_views(html):
    for needle in ['id="ticket_id"', 'id="product_category"', 'id="product_age_days"', 'id="complaint_text"',
                   'id="view-classify"', 'id="view-bulk"', 'id="view-dashboard"', 'id="file"',
                   '"alert"', "/classify", "/metrics", "/scorecard", "/health"]:
        assert needle in html, needle
    for product in ["washing_machine", "refrigerator", "dishwasher", "oven_or_cooktop", "air_conditioner", "water_heater", "microwave"]:
        assert f'value="{product}"' in html


def test_how_it_works_view(html):
    for needle in ['id="view-how"', 'id="tab-how"', 'id="how-text"', 'id="how-next"', 'id="how-back"', 'id="how-all"', "/explain"]:
        assert needle in html, needle
    for title in ["Clean-up", "Safety scan", "Sentence split", "Sentence classifier", "Symptoms found", "Rule ladder"]:
        assert title in html, title
    for field in ["how-product", "how-age", "how-text"]:
        assert f'for="{field}"' in html


def test_evals_and_dashboard_views(html):
    for needle in ['id="view-evals"', 'id="tab-evals"', 'id="sc-schemes"', 'id="sc-classes"', 'id="sc-cm"', 'id="ev-family"',
                   'id="ev-cov"', 'id="ev-slice-product"', 'id="sc-gates"', 'id="dash-kpis"', 'id="dash-series"', 'id="dash-p95"',
                   'id="dash-gauge"', 'id="dash-heat"', 'id="dash-cats"', 'id="dash-paths"', 'id="dash-recent"', 'id="dash-quality"', 'id="dash-conf"', 'id="how-marked"', 'id="how-rail"', 'id="how-diff"']:
        assert needle in html, needle


def test_all_six_business_rules_are_explained(html):
    for rule in ["C1", "C2", "C3", "C4", "C5", "C6"]:
        assert f'id: "{rule}"' in html


def test_no_external_resources(html):
    urls = set(re.findall(r"https?://[^\s\"'<>)]+", html))
    assert urls <= {"http://www.w3.org/2000/svg", "http://localhost:8000/classify"}, urls
    assert "@import" not in html and "<link" not in html and "fonts.googleapis" not in html


def test_no_innerhtml(html):
    assert "innerHTML" not in html and "document.write" not in html


def test_labels_bound_to_inputs(html):
    for field in ["ticket_id", "product_category", "product_age_days", "complaint_text", "file"]:
        assert f'for="{field}"' in html


def test_design_tokens(html):
    for token in ["--bg: #0a0a0a", "--accent: #ff6b35", "--danger: #ff4d4f", "--font-mono"]:
        assert token in html, token
    assert "bruviti" not in html.lower()
