from src.build_data import process_csv
from src.preprocessing import load_and_prepare_data
from src.train import train_model
from pathlib import Path
def main():
    # process_csv("data/unprocessed-data/phishing_url.csv", "data/preprocessed/non_legit.csv")
    # remove_zero_bert_rows("data/preprocessed/non_legit.csv")
    # process_csv("data/unprocessed-data/non_phishing_url.csv", "data/preprocessed/legit.csv")
    X, y = load_and_prepare_data(
    "data/preprocessed/legit.csv",
    "data/preprocessed/non_legit.csv"
)

    model, scaler = train_model(X, y, str(Path.cwd() / "output"))
    
if __name__ == "__main__":
    main()