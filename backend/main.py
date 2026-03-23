from src.preprocessed import process_csv,remove_zero_bert_rows
def main():
    # process_csv("data/unprocessed-data/phishing_url.csv", "data/preprocessed/non_legit.csv")
    # remove_zero_bert_rows("data/preprocessed/non_legit.csv")
    process_csv("data/unprocessed-data/non_phishing_url.csv", "data/preprocessed/legit.csv")
    remove_zero_bert_rows("data/preprocessed/legit.csv")

    
if __name__ == "__main__":
    main()