import numpy as np
from load_model import load_model


def test_model():
    model, scaler = load_model()

    sample = {
    "url_length": 36,
    "num_dots": 2,
    "num_hyphens": 0,
    "num_digits": 0,
    "has_at_symbol": 0,
    "has_ip": 0,
    "num_subdomains": 1,
    "entropy": 3.8145364592347777,
    "num_forms": 1,
    "num_inputs": 8,
    "has_password": 0,
    "num_iframes": 0,
    "keyword_score": 1.0,
    "bert_score": 0.7065286636352539
}

    sample["url_complexity"] = (
        sample["num_dots"] +
        sample["num_hyphens"] +
        sample["num_digits"]
    )

    sample["form_risk"] = (
        sample["num_forms"] +
        sample["has_password"]
    )

    sample["structure_score"] = (
        sample["num_inputs"] +
        sample["num_iframes"]
    )

    feature_order = [
        "url_length", "num_dots", "num_hyphens", "num_digits",
        "has_at_symbol", "has_ip", "num_subdomains", "entropy",
        "num_forms", "num_inputs", "has_password", "num_iframes",
        "keyword_score", "bert_score",
        "url_complexity", "form_risk", "structure_score"
    ]

    features = np.array([sample[col] for col in feature_order]).reshape(1, -1)

    features = scaler.transform(features)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    print("\nPrediction:", "PHISHING " if prediction == 1 else "LEGIT ")
    print("Confidence:", probability)


if __name__ == "__main__":
    test_model()