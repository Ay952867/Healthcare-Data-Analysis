import os
import joblib
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 🔹 Dictionary of diseases and their precautions
DISEASE_PRECAUTIONS = {
    "Flu": "Rest, stay hydrated, and take paracetamol for fever.",
    "Migraine": "Avoid bright lights and stay hydrated.",
    "Common Cold": "Rest, steam inhalation, and drink warm fluids.",
    "Measles": "Stay isolated and consult a doctor.",
    "Food Poisoning": "Drink ORS, rest, and avoid solid foods initially.",
    "Unknown": "Consult a healthcare professional for further diagnosis."
}


# 🔹 Symptom Model Class
class SymptomModel:
    def __init__(self, data_path="dataset.csv"):
        self.data_path = data_path
        self.model_path = "models/disease_model.pkl"
        self.scaler_path = "models/scaler.pkl"
        self.features_path = "models/features.pkl"
        self.model = None
        self.scaler = None
        self.features = []

    def train_if_needed(self):
        """Train the model if no saved model exists, otherwise load it."""
        # Load if already trained
        if (
            os.path.exists(self.model_path)
            and os.path.exists(self.scaler_path)
            and os.path.exists(self.features_path)
        ):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.features = joblib.load(self.features_path)
            print("✅ Model, Scaler, and Features loaded successfully.")
            return

        print("⚙️ Training new model...")

        # Load dataset
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        df = pd.read_csv(self.data_path)

        if "disease" not in df.columns:
            raise ValueError("Dataset must contain a 'disease' column.")

        X = df.drop(columns=["disease"])
        y = df["disease"]
        self.features = X.columns.tolist()

        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Save model, scaler, and features
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.features, self.features_path)

        print("✅ Model trained and saved successfully.")

    def predict_from_symptoms(self, symptoms: List[str]):
        """Predict disease probabilities from a list of symptoms."""
        if self.model is None or self.scaler is None or not self.features:
            self.train_if_needed()

        # Create binary feature vector
        symptom_vector = np.zeros(len(self.features))
        for s in symptoms:
            s_clean = s.lower().strip().replace(" ", "_")
            if s_clean in self.features:
                symptom_vector[self.features.index(s_clean)] = 1

        # If vector is empty or invalid
        if len(self.features) == 0:
            raise ValueError("No features found. Model might not be trained properly.")

        scaled = self.scaler.transform(symptom_vector.reshape(1, -1))
        probs = self.model.predict_proba(scaled)[0]
        classes = self.model.classes_

        predictions = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        return predictions


# 🔹 Helper function for precautions
def get_precautions(disease: str) -> str:
    return DISEASE_PRECAUTIONS.get(disease, DISEASE_PRECAUTIONS["Unknown"])


# 🔹 Singleton instance for Streamlit app
_model_instance: SymptomModel = None


def get_model_instance() -> SymptomModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = SymptomModel()
        _model_instance.train_if_needed()
    return _model_instance


# 🔹 Streamlit helper function
def predict(symptoms: List[str]):
    """Predict diseases and return formatted results."""
    model = get_model_instance()
    preds = model.predict_from_symptoms(symptoms)

    results = []
    for disease, prob in preds:
        results.append({
            "disease": disease,
            "probability": float(prob),
            "precautions": get_precautions(disease)
        })

    return results
