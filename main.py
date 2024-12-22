import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the data from the SQLite database
def load_data_from_db(db_file):
    con = sqlite3.connect(db_file)
    df = pd.read_sql_query("SELECT * FROM tablename", con)
    con.close()
    print(db_file, df.columns.tolist())
    return df

# Step 2: Preprocess the data
def preprocess_data(df):
    # Remove unnecessary columns
    df = df.drop(['id', 'model_a', 'model_b'], axis=1)

    # Define the target: 1 if model_a wins, 0 if model_b wins, 2 if tie
    target = 'winner'

    def determine_winner(row):
        if row['winner_model_a'] == 1:
            return 1
        elif row['winner_model_b'] == 1:
            return 0
        else:
            return 2

    df[target] = df.apply(determine_winner, axis=1)

    # Convert categorical text data into numerical format
    le_prompt = LabelEncoder()
    le_response_a = LabelEncoder()
    le_response_b = LabelEncoder()

    df['prompt_encoded'] = le_prompt.fit_transform(df['prompt'])
    df['response_a_encoded'] = le_response_a.fit_transform(df['response_a'])
    df['response_b_encoded'] = le_response_b.fit_transform(df['response_b'])

    X = df[['prompt_encoded', 'response_a_encoded', 'response_b_encoded']]
    y = df[target]

    return X, y, le_prompt, le_response_a, le_response_b

# Step 3: Train the model using XGBoost
def train_model(X, y):
    # Split the data to train and test for evaluation
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

    # Construct an XGBoost DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_eval, label=y_eval)

    # Set XGBoost parameters
    params = {
        'objective': 'multi:softprob',  # Multiclass classification
        'eval_metric': 'mlogloss',
        'num_class': 3  # We have three classes (model_a wins, model_b wins, tie)
    }

    # Train the model
    model = xgb.train(params, dtrain, num_boost_round=100)

    # Evaluate the model
    preds = model.predict(dtest)
    best_preds = np.asarray([np.argmax(line) for line in preds])
    accuracy = accuracy_score(y_eval, best_preds)
    print(f'Model accuracy: {accuracy}')

    return model

#####
# Step 4: Load test set and make predictions
def predict_test_set(model, le_prompt, le_response_a, le_response_b, db_file):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    query = "SELECT * FROM tablename"  # Replace 'test_table' with your actual table name
    test_df = pd.read_sql_query(query, conn)
    conn.close()

    test_df = test_df.drop(['id'], axis=1)

    # Encode the test dataframe using the same LabelEncoders
    test_df['prompt_encoded'] = le_prompt.transform(test_df['prompt'])
    test_df['response_a_encoded'] = le_response_a.transform(test_df['response_a'])
    test_df['response_b_encoded'] = le_response_b.transform(test_df['response_b'])

    X_test = test_df[['prompt_encoded', 'response_a_encoded', 'response_b_encoded']]
    dtest = xgb.DMatrix(X_test)

    # Make predictions
    preds = model.predict(dtest)
    best_preds = [np.argmax(pred) for pred in preds]

    # Map integers back to their respective preferences
    test_df['predicted_winner'] = best_preds
    return test_df

if __name__ == "__main__":
    # Load, preprocess, and split the data
    db_file = 'training.db'
    df = load_data_from_db(db_file)
    X, y, le_prompt, le_response_a, le_response_b = preprocess_data(df)

    # Train the model
    model = train_model(X, y)

    # Predict on the test set
    predictions = predict_test_set(model, le_prompt, le_response_a, le_response_b, 'testing.db')

    # Display predictions
    print(predictions[['id', 'predicted_winner']])
#####
