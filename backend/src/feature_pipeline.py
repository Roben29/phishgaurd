import requests
import re
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from collections import Counter
from transformers import pipeline
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

phishing_model = pipeline(
    "text-classification",
    model="cybersectony/phishing-email-detection-distilbert_v2.1"
)

session = requests.Session()
retry = Retry(total=2, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

phishing_keywords = [
    "verify", "password", "login", "urgent", "account",
    "bank", "update", "confirm", "secure", "click",
    "limited", "suspended", "alert", "reset", "immediately"
]

def url_entropy(url):
    prob = [n / len(url) for n in Counter(url).values()]
    return -sum(p * np.log2(p) for p in prob)


def extract_url_features(url):
    parsed = urlparse(url)

    return {
        "url_length": len(url),
        "num_dots": url.count('.'),
        "num_hyphens": url.count('-'),
        "num_digits": sum(c.isdigit() for c in url),
        "has_at_symbol": int('@' in url),
        "has_ip": int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))),
        "num_subdomains": max(len(parsed.netloc.split('.')) - 2, 0),
        "entropy": url_entropy(url),
    }

def scrape(url):
    try:
        res = session.get(url, timeout=5)
        if res.status_code >= 400:
            return None, ""

        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ").lower()
        text = " ".join(text.split())

        return soup, text[:800]

    except Exception as e:
        logger.warning(f"Scraping failed: {url}")
        return None, ""

def extract_html_features(soup):
    if soup is None:
        return {
            "num_forms": 0,
            "num_inputs": 0,
            "has_password": 0,
            "num_iframes": 0
        }

    return {
        "num_forms": len(soup.find_all("form")),
        "num_inputs": len(soup.find_all("input")),
        "has_password": int(bool(soup.find("input", {"type": "password"}))),
        "num_iframes": len(soup.find_all("iframe"))
    }

def keyword_score(text):
    if not text:
        return 0
    return sum(word in text for word in phishing_keywords) / len(phishing_keywords)


def bert_score(text):
    if not text or len(text) < 20:
        return 0

    try:
        result = phishing_model(text[:512])[0]
        return result["score"] if "phishing" in result["label"].lower() else 1 - result["score"]
    except:
        return 0

FEATURE_ORDER = [
    "url_length", "num_dots", "num_hyphens", "num_digits",
    "has_at_symbol", "has_ip", "num_subdomains", "entropy",
    "num_forms", "num_inputs", "has_password", "num_iframes",
    "keyword_score", "bert_score",
    "url_complexity", "form_risk", "structure_score"
]

def build_features(url):
    features = {}

    features.update(extract_url_features(url))
    soup, text = scrape(url)
    features.update(extract_html_features(soup))

    features["keyword_score"] = keyword_score(text)
    features["bert_score"] = bert_score(text)

    features["url_complexity"] = (
        features["num_dots"] +
        features["num_hyphens"] +
        features["num_digits"]
    )

    features["form_risk"] = (
        features["num_forms"] +
        features["has_password"]
    )

    features["structure_score"] = (
        features["num_inputs"] +
        features["num_iframes"]
    )

    vector = [features.get(f, 0) for f in FEATURE_ORDER]

    return np.array(vector).reshape(1, -1)