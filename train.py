# =============================================
# Random Forest on Iris Dataset (Clean + autolog)
# =============================================

import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Enable autolog → automatically logs params, metrics, model + signature!
mlflow.autolog(log_models=True, log_input_examples=True, silent=True)  # silent=True removes extra warnings

# 1. Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["species"] = pd.Categorical.from_codes(y, target_names)

print("Dataset shape:", X.shape)
print(df.head())
print("\nClass distribution:\n", df["species"].value_counts())

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# 3. Set experiment & start run
mlflow.set_experiment("Iris Random Forest Classification")

with mlflow.start_run(run_name="RF_n300_depth10") as run:

    # Custom tags
    mlflow.set_tag("model_type", "RandomForestClassifier")
    mlflow.set_tag("dataset", "Iris")
    mlflow.set_tag("project", "mlops-demo")

    # 4. Train model (autolog will capture everything)
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=47,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True
    )
    rf.fit(X_train, y_train)

    # 5. Predictions & evaluation
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

    # 6. Confusion Matrix (autolog doesn't do this → manual)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    # 7. Log train/test datasets properly (autolog doesn't do this with context)
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df["species"] = pd.Categorical.from_codes(y_train, target_names)
    test_df  = pd.DataFrame(X_test,  columns=feature_names)
    test_df["species"]  = pd.Categorical.from_codes(y_test,  target_names)

    mlflow.log_input(mlflow.data.from_pandas(train_df, targets="species"), context="training")
    mlflow.log_input(mlflow.data.from_pandas(test_df,  targets="species"), context="test")

    # Optional: save CSVs for easy download
    train_df.to_csv("train_set.csv", index=False)
    test_df.to_csv("test_set.csv", index=False)
    mlflow.log_artifact("train_set.csv")
    mlflow.log_artifact("test_set.csv")

    # 8. Log the source script
    mlflow.log_artifact(__file__)

    print(f"\nRun completed! View at: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.get_experiment_by_name('Iris Random Forest Classification').experiment_id}/runs/{run.info.run_id}")