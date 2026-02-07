import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from ocr import extract_text   # OCR logic (Tesseract)

app = Flask(__name__)
CORS(app)

# ---------------- LOAD MODEL & DATA ----------------

# Load trained ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load dataset (for side effects lookup)
df = pd.read_csv("drug_dataset.csv")

# ---------------- API ROUTE ----------------

@app.route("/predict", methods=["POST"])
def predict():

    # 1. Check image upload
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]

    try:
        # 2. OCR: extract text from image
        extracted_text = extract_text(image)
        extracted_text_lower = extracted_text.lower()

        # 3. Identify drug name using rule-based NLP
        detected_drug = None
        for drug in df["drug_name"].unique():
            if drug.lower() in extracted_text_lower:
                detected_drug = drug
                break

        if not detected_drug:
            return jsonify({
                "drug": "Unknown",
                "risk": "Unknown",
                "side_effects": "Drug not found in dataset"
            })

        # 4. Get latest drug row (example logic)
        drug_row = df[df["drug_name"] == detected_drug].iloc[-1]

        # 5. Prepare ML features
        features = [[
            drug_row["dosage"],
            drug_row["max_safe_dosage"],
            drug_row["interaction_flag"],
            drug_row["side_effect_score"]
        ]]

        # 6. Predict risk using ML model
        prediction = model.predict(features)
        risk_label = le.inverse_transform(prediction)[0]

        # 7. Side effects (rule-based from dataset)
        side_effects = drug_row.get("common_side_effects", "Refer doctor")

        # 8. Final response
        return jsonify({
            "drug": detected_drug,
            "risk": risk_label,
            "side_effects": side_effects
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True)
