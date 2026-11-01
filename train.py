import pandas as pd, joblib, sqlite3
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Load and check the data
df = pd.read_csv("respiratory_patients.csv")
print("Rows:", len(df), "| Missing values:", df.isna().sum().sum(), "| Duplicate rows:", df.drop(columns="Patient Id").duplicated().sum())
df.to_sql("patients", sqlite3.connect("pulmo.db"), if_exists="replace", index=False)  # all patients go to SQL

# 2. Preprocess: text columns (Smoking Status, Wheezing, ...) become 0/1 columns; numbers stay as they are
X, y = df.drop(columns=["Patient Id", "Risk Level"]), df["Risk Level"]
text_cols = X.select_dtypes(exclude="number").columns.tolist()
prep = ColumnTransformer([("text", OneHotEncoder(handle_unknown="ignore"), text_cols)], remainder="passthrough")
model = Pipeline([("prep", prep), ("rf", RandomForestClassifier(random_state=42))])

# 3. Split, validate, test
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("5-fold CV accuracy (train set):", cross_val_score(model, X_tr, y_tr, cv=5).mean())
model.fit(X_tr, y_tr)
print(classification_report(y_te, model.predict(X_te)))
joblib.dump(model, "model.pkl")
