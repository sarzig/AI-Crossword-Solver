import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from clue_classification_and_processing.clue_features import add_features, add_profession

# Feature methods

# Load the Excel file
df = pd.read_excel("manually classified clues.xlsx")

# Filter rows where "Class (Manual)" is not blank
clues = df[df["Class (Manual)"].notna() & (df["Class (Manual)"].astype(str).str.strip() != "")][["Clue", "Word", "Class (Manual)"]]

# add features
clues = add_features(clues)


# Target
y = clues["Class (Manual)"]

# You can select all other columns except target as features
X = clues.drop(columns=["Class (Manual)"])

text_features = ["Clue"]

numeric_features = [col for col in X.columns if col not in text_features and col != "Word"]

# 2. ColumnTransformer without "Word"
preprocessor = ColumnTransformer([
    ("tfidf_clue", TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=1000), "Clue"),
    ("numeric_features", FunctionTransformer(lambda x: x[numeric_features], validate=False), numeric_features)
])

pipeline = Pipeline([
    ("features", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))