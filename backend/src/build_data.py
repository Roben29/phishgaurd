import csv
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
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
retry = Retry(total=3, backoff_factor=1)
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

phishing_keywords = [
    "verify", "password", "login", "urgent", "account",
    "bank", "update", "confirm", "secure", "click",
    "limited", "suspended", "alert", "reset", "immediately"
]

def url_entropy(url):
    prob = [n_x / len(url) for x, n_x in Counter(url).items()]
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

def scrape_html(url):
    try:
        res = session.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept-Language": "en-US,en;q=0.9"
            }
        )

        if res.status_code >= 400:
            return None, ""

        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ").lower()
        text = " ".join(text.split())

        return soup, text[:1000]

    except Exception as e:
        logger.warning(f"Scraping failed: {url} | {e}")
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
        "has_password": int(len(soup.find_all("input", {"type": "password"})) > 0),
        "num_iframes": len(soup.find_all("iframe"))
    }

def keyword_score(text):
    if not text:
        return 0

    count = sum(1 for word in phishing_keywords if word in text)
    return count / len(phishing_keywords)

def bert_score(text):
    if not text or len(text) < 20:
        return 0

    try:
        result = phishing_model(text[:512])[0]
        prob = result["score"] if "phishing" in result["label"].lower() else 1 - result["score"]
        return prob

    except Exception as e:
        logger.warning(f"BERT failed: {e}")
        return 0

def extract_features(url):
    features = {}

    try:
        features.update(extract_url_features(url))

        soup, text = scrape_html(url)

        features.update(extract_html_features(soup))

        features["keyword_score"] = keyword_score(text)
        features["bert_score"] = bert_score(text)

    except Exception as e:
        logger.error(f"Feature extraction failed: {url} | {e}")

    return features

def normalize_url(url):
    url = url.strip()

    if not url or url.lower() == "url":
        return None

    if not url.startswith("http"):
        url = "http://" + url

    return url

FIELDS = [
    "url_length", "num_dots", "num_hyphens", "num_digits",
    "has_at_symbol", "has_ip", "num_subdomains", "entropy",
    "num_forms", "num_inputs", "has_password", "num_iframes",
    "keyword_score", "bert_score"
]

def process_csv(input_file, output_file):
    total = 0

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", newline='', encoding="utf-8") as outfile:

        reader = csv.reader(infile)
        next(reader, None)

        writer = csv.DictWriter(outfile, fieldnames=FIELDS)
        writer.writeheader()

        for row in reader:
            url = normalize_url(row[0])

            if not url:
                continue

            total += 1

            features = extract_features(url)

            row_data = {key: features.get(key, 0) for key in FIELDS}

            writer.writerow(row_data)

            if total % 20 == 0:
                logger.info(f"Processed {total} URLs")

    logger.info("Feature extraction completed")