"""Train RF with sampled data to avoid OOM."""
import sys, os
sys.path.insert(0, "F:\\WEATHER_MINING_PROJECT")
os.chdir("F:\\WEATHER_MINING_PROJECT")

import pandas as pd, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from src.data.loader import load_config, resolve_path

LABEL_MAP = {
    "Clear": "Clear",
    "Partly Cloudy": "Cloudy", "Mostly Cloudy": "Cloudy", "Overcast": "Cloudy",
    "Dry": "Cloudy", "Dry and Partly Cloudy": "Cloudy", "Dry and Mostly Cloudy": "Cloudy",
    "Humid and Partly Cloudy": "Cloudy", "Humid and Mostly Cloudy": "Cloudy",
    "Humid and Overcast": "Cloudy",
    "Rain": "Rain", "Drizzle": "Rain", "Light Rain": "Rain",
    "Foggy": "Foggy", "Breezy and Foggy": "Foggy",
    "Breezy": "Windy", "Breezy and Mostly Cloudy": "Windy", "Breezy and Overcast": "Windy",
    "Breezy and Partly Cloudy": "Windy", "Windy and Mostly Cloudy": "Windy",
    "Windy and Overcast": "Windy", "Windy and Partly Cloudy": "Windy",
    "Windy": "Windy", "Windy and Foggy": "Windy",
    "Dangerously Windy and Partly Cloudy": "Windy",
}

config = load_config()
df = pd.read_parquet(resolve_path(config["data"]["cleaned_parquet"]))
df["Weather_Group"] = df["Summary"].map(LABEL_MAP).fillna("Cloudy")

# Stratified SAMPLE 20K to fit in memory
df_sample = df.groupby("Weather_Group", group_keys=False).apply(
    lambda x: x.sample(min(len(x), 4000), random_state=42)
)
print(f"Sampled: {len(df_sample)} rows")
print(df_sample["Weather_Group"].value_counts())

FEATURES = ["Temperature (C)", "Apparent Temperature (C)", "Humidity",
            "Wind Speed (km/h)", "Wind Bearing (degrees)",
            "Visibility (km)", "Pressure (millibars)"]
X = df_sample[FEATURES].values
le = LabelEncoder()
y = le.fit_transform(df_sample["Weather_Group"].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Training RF (n_jobs=1)...")
rf = RandomForestClassifier(
    n_estimators=150, max_depth=18, random_state=42,
    class_weight="balanced", n_jobs=1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
f1 = f1_score(y_test, y_pred, average="macro")
acc = accuracy_score(y_test, y_pred)
print(f"F1-macro: {f1:.4f} | Accuracy: {acc:.4f}")

# Save
model_dir = resolve_path(config["outputs"]["models_dir"])
os.makedirs(model_dir, exist_ok=True)
joblib.dump(rf, os.path.join(model_dir, "best_classifier.joblib"))
joblib.dump(le, os.path.join(model_dir, "label_encoder.joblib"))

results = pd.DataFrame([{"Model":"Random Forest","F1_macro":round(f1,4),"Accuracy":round(acc,4)}])
tables_dir = resolve_path(config["outputs"]["tables_dir"])
results.to_csv(os.path.join(tables_dir, "classification_results.csv"), index=False)

print(f"\n DONE! F1-macro = {f1:.4f}")
