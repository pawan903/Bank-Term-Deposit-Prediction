import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay

# Update this path to where bank.csv is saved on your computer
DATA_PATH = r"C:\Users\VICTUS\Downloads\bank.csv"
df = pd.read_csv(DATA_PATH)
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()
df = df.drop_duplicates().copy()
print("Dataset shape:", df.shape)
print(df.head())
print(df.info())
print("Missing values:\\n", df.isnull().sum())
print("Target distribution:\\n", df["deposit"].value_counts())

# EDA charts
for col in ["deposit", "age", "job", "education", "housing", "loan"]:
    plt.figure(figsize=(8, 4))
    if df[col].nunique() < 15:
        df[col].value_counts().plot(kind="bar")
        plt.xticks(rotation=30, ha="right")
    else:
        df[col].plot(kind="hist", bins=25, edgecolor="black")
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()

# Predict before the call: duration is excluded because it is only known after the call.
y = (df["deposit"].str.lower() == "yes").astype(int)
X = df.drop(columns=["deposit", "duration"], errors="ignore")
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=np.number).columns.tolist()
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
}
rows = []
trained = {}
for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred, zero_division=0), "Recall": recall_score(y_test, pred, zero_division=0), "F1-score": f1_score(y_test, pred, zero_division=0), "ROC-AUC": roc_auc_score(y_test, prob)})
    trained[name] = (pipe, pred, prob)
results = pd.DataFrame(rows).sort_values("F1-score", ascending=False)
print("\\nModel comparison:\\n", results.to_string(index=False))
best_name = results.iloc[0]["Model"]
best_pipe, best_pred, best_prob = trained[best_name]
print("Best model by F1-score:", best_name)
print(classification_report(y_test, best_pred, target_names=["No deposit", "Deposit"], zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test, best_pred, display_labels=["No", "Yes"])
plt.tight_layout(); plt.show()
RocCurveDisplay.from_predictions(y_test, best_prob)
plt.tight_layout(); plt.show()
results.to_csv("model_performance_results.csv", index=False)
print("Results saved to model_performance_results.csv")
