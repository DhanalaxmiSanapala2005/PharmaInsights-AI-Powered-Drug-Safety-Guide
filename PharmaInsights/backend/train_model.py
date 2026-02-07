import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("drug_dataset.csv")

X = df[["dosage", "max_safe_dosage", "interaction_flag", "side_effect_score"]]
y = df["risk"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = LogisticRegression(max_iter=1000)
model.fit(X, y_encoded)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("✅ Model trained and saved")
