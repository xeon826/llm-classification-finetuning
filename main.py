import sqlite3
from scipy.sparse import hstack
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


# Step 1: Load the data from the SQLite database
def load_data_from_db(db_file):
    con = sqlite3.connect(db_file)
    df = pd.read_sql_query("SELECT * FROM tablename", con)
    con.close()
    print(db_file, df.columns.tolist())
    return df


# Step 2: Preprocess the data
def preprocess_data_with_tfidf(df):
    vectorizer_prompt = TfidfVectorizer(max_features=2000)
    vectorizer_response_a = TfidfVectorizer(max_features=2000)
    vectorizer_response_b = TfidfVectorizer(max_features=2000)

    prompt_tfidf = vectorizer_prompt.fit_transform(df["prompt"])
    response_a_tfidf = vectorizer_response_a.fit_transform(df["response_a"])
    response_b_tfidf = vectorizer_response_b.fit_transform(df["response_b"])

    # Use hstack for sparse matrix concatenation
    X = hstack([prompt_tfidf, response_a_tfidf, response_b_tfidf])

    def determine_winner(row):
        if row["winner_model_a"] == 1:
            return 1
        elif row["winner_model_b"] == 1:
            return 0
        else:
            return 2

    y = df.apply(determine_winner, axis=1)
    return X, y, vectorizer_prompt, vectorizer_response_a, vectorizer_response_b


# Step 3: Train the model using XGBoost
def train_model(X, y):
    # Split the data to train and test for evaluation
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Construct an XGBoost DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_eval, label=y_eval)

    # Set XGBoost parameters
    params = {
        "objective": "multi:softprob",  # Multiclass classification
        "eval_metric": "mlogloss",
        "num_class": 3,  # We have three classes (model_a wins, model_b wins, tie)
    }

    # Train the model
    model = xgb.train(params, dtrain, num_boost_round=100)

    # Evaluate the model
    preds = model.predict(dtest)
    best_preds = np.asarray([np.argmax(line) for line in preds])
    accuracy = accuracy_score(y_eval, best_preds)
    print(f"Model accuracy: {accuracy}")

    return model


#####
# Step 4: Load test set and make predictions
def predict_test_set_with_tfidf(
    model, vectorizer_prompt, vectorizer_response_a, vectorizer_response_b, db_file
):
    conn = sqlite3.connect(db_file)
    test_df = pd.read_sql_query("SELECT * FROM tablename", conn)
    conn.close()

    prompt_tfidf = vectorizer_prompt.transform(test_df["prompt"])
    response_a_tfidf = vectorizer_response_a.transform(test_df["response_a"])
    response_b_tfidf = vectorizer_response_b.transform(test_df["response_b"])

    # Use hstack for sparse matrix concatenation
    X_test = hstack([prompt_tfidf, response_a_tfidf, response_b_tfidf])
    dtest = xgb.DMatrix(X_test)

    preds = model.predict(dtest)
    best_preds = [np.argmax(pred) for pred in preds]

    test_df["predicted_winner"] = best_preds
    return test_df


if __name__ == "__main__":
    # Load, preprocess, and split the data
    db_file = "training.db"
    df = load_data_from_db(db_file)
    X, y, le_prompt, le_response_a, le_response_b = preprocess_data_with_tfidf(df)

    # Train the model
    model = train_model(X, y)

    # Predict on the test set
    predictions = predict_test_set_with_tfidf(
        model, le_prompt, le_response_a, le_response_b, "testing.db"
    )

    # Display predictions
    print(predictions[["id", "predicted_winner"]])
#####
