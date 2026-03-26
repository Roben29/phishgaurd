from src.load_model import load_model
from src.feature_pipeline import build_features

def predict_url(url, threshold=0.5):
    model, scaler = load_model()

    features = build_features(url)
    features = scaler.transform(features)

    # Get probabilities
    probs = model.predict_proba(features)[0]
    legit_prob = probs[0]
    phishing_prob = probs[1]

    # Decision based on threshold
    if phishing_prob > threshold:
        prediction = "PHISHING 🚨"
        confidence = phishing_prob
    else:
        prediction = "LEGIT ✅"
        confidence = legit_prob

    return {
        "url": url,
        "prediction": prediction,
        "confidence": float(confidence),
        "probabilities": {
            "legit": float(legit_prob),
            "phishing": float(phishing_prob)
        }
    }