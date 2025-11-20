# src/model_building.py
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from utils import load_params, logger
import os

def main():
    params = load_params()['model_building']
    
    train_df = pd.read_csv("data/features/train_bow.csv")
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values

    model = GradientBoostingClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        random_state=params['random_state']
    )
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    logger.info("Model trained and saved successfully!")

if __name__ == "__main__":
    main()