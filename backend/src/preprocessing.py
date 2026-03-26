import pandas as pd

def load_and_prepare_data(legit_path, non_legit_path):
    legit = pd.read_csv(legit_path)
    non_legit = pd.read_csv(non_legit_path)

    legit["label"] = 0
    non_legit["label"] = 1

    df = pd.concat([legit, non_legit], ignore_index=True)

    # Feature Engineering
    df["url_complexity"] = df["num_dots"] + df["num_hyphens"] + df["num_digits"]
    df["form_risk"] = df["num_forms"] + df["has_password"]
    df["structure_score"] = df["num_inputs"] + df["num_iframes"]

    df = df.fillna(df.median())

    X = df.drop("label", axis=1)
    y = df["label"]

    return X, y