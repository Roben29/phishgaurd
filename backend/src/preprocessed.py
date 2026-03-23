import csv
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from urllib.parse import urlparse
from collections import Counter
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

intent_model = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def url_entropy(url):
    prob = [n_x / len(url) for x, n_x in Counter(url).items()]
    entropy = -sum(p * np.log2(p) for p in prob)
    return entropy

def extract_url_features(url):
    logger.debug(f"Extracting URL features: {url}")

    parsed = urlparse(url)

    return {
        "url_length": len(url),
        "num_dots": url.count('.'),
        "num_hyphens": url.count('-'),
        "num_digits": sum(c.isdigit() for c in url),
        "has_at_symbol": int('@' in url),
        "has_ip": int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))),
        "num_subdomains": len(parsed.netloc.split('.')) - 2,
        "entropy": url_entropy(url)
    }

def scrape_html(url):
    try:
        logger.info(f"Scraping started: {url}")

        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ")

        logger.info(f"Scraping success: {url}")
        return soup, text.strip()

    except Exception as e:
        logger.error(f"Scraping failed: {url} | Error: {e}")
        return None, None

def extract_html_features(soup):
    if soup is None:
        logger.warning("Soup is None, returning default HTML features")
        return {
            "num_forms": 0,
            "num_inputs": 0,
            "has_password": 0,
            "num_iframes": 0
        }

    forms = soup.find_all("form")
    inputs = soup.find_all("input")
    iframes = soup.find_all("iframe")

    password_fields = soup.find_all("input", {"type": "password"})

    logger.debug(f"HTML features extracted: forms={len(forms)}, inputs={len(inputs)}")

    return {
        "num_forms": len(forms),
        "num_inputs": len(inputs),
        "has_password": int(len(password_fields) > 0),
        "num_iframes": len(iframes)
    }

phishing_keywords = [
    "login","verify","account","bank","password","update","secure","confirm","urgent","payment",
    "username","credential","identity","profile","email","userid","authenticate","validation","registered",
    "immediately","suspended","limited","expire","restricted","warning","alert","critical","required","action",
    "billing","transaction","refund","invoice","credit","debit","wallet","wire","transfer","checkout",
    "protection","encrypted","official","compliance","authorized","trusted","certificate","2fa","otp","verification",
    "click","submit","access","unlock","reactivate","respond","review","complete","proceed","resolve",
    "support","helpdesk","notification","service","admin","customer","prize","reward","gift",
    "redirect","portal","token","session","reset","recovery","signin","logon","webmail",
    "free","winner","won","congratulations","selected","exclusive","offer","deal","discount","promo",
    "coupon","cashback","bonus","claim","redeem","giveaway","jackpot","lottery","sweepstakes","eligible",
    "limited-time","special","savings","trial","subscription","voucher","points","rebate","unlock","granted",
]

def keyword_score(text):
    text = text.lower()
    score = sum(word in text for word in phishing_keywords)
    logger.debug(f"Keyword score: {score}")
    return score

def bert_intent_score(text):
    if not text:
        logger.warning("Empty text for BERT scoring")
        return 0

    try:
        text = text[:512]
        result = intent_model(text)[0]

        logger.debug(f"BERT result: {result}")

        return result["score"]

    except Exception as e:
        logger.error(f"BERT failed | Error: {e}")
        return 0

def extract_features(url):
    logger.info(f"Processing URL: {url}")

    features = {}

    try:
        features.update(extract_url_features(url))

        soup, text = scrape_html(url)

        features.update(extract_html_features(soup))

        features["keyword_score"] = keyword_score(text if text else "")
        features["bert_score"] = bert_intent_score(text)

        logger.info(f"Completed URL: {url}")

    except Exception as e:
        logger.error(f"Feature extraction failed: {url} | Error: {e}")

    return features

def normalize_url(url):
    url = url.strip()

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url

    return url

def process_csv(input_file, output_file):
    logger.info("CSV processing started")

    total = 0

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", newline='', encoding="utf-8") as outfile:

        reader = csv.reader(infile)
        writer = None

        for row in reader:
            total += 1
            url = normalize_url(row[0])

            try:
                features = extract_features(url)

                if writer is None:
                    writer = csv.DictWriter(outfile, fieldnames=features.keys())
                    writer.writeheader()

                writer.writerow(features)

            except Exception as e:
                logger.error(f"Row failed: {url} | Error: {e}")

            if total % 50 == 0:
                logger.info(f"Processed {total} URLs")

    logger.info("Feature extraction completed")

def remove_zero_bert_rows(file_path):
    import csv
    import os

    temp_file = file_path + ".tmp"

    kept = 0
    removed = 0

    with open(file_path, "r", encoding="utf-8") as infile, \
         open(temp_file, "w", newline='', encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)

        writer.writeheader()

        for row in reader:
            try:
                bert_score = float(row.get("bert_score", 0))

                if bert_score != 0:
                    writer.writerow(row)
                    kept += 1
                else:
                    removed += 1

            except Exception as e:
                removed += 1

    os.replace(temp_file, file_path)

    logger.info(f"BERT filter applied → Kept: {kept}, Removed: {removed}")