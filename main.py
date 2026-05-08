from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import uvicorn
import shap
import numpy as np

app = FastAPI(
    title="Credit Scoring API",
    description="Low-latency inference API with SHAP explainability for credit risk tier prediction.",
    version="1.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
try:
    model = joblib.load("xgb_top15_undersampled_model.pkl")
    top_features = joblib.load("top_15_features.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    
    # Initialize the SHAP explainer once when the app starts
    explainer = shap.TreeExplainer(model)
except Exception as e:
    raise RuntimeError(f"Failed to load artifacts or initialize explainer. Error: {e}")

# Strict payload definition matching ONLY the required 15 features
class ApplicantData(BaseModel):
    enq_L3m: int = Field(..., description="Inquiries in the last 3 months")
    Age_Oldest_TL: int = Field(..., description="Age of oldest tradeline")
    num_times_delinquent: int = Field(..., description="Total delinquencies")
    num_std_12mts: int = Field(..., description="Standard payments in last 12 months")
    pct_PL_enq_L6m_of_ever: float = Field(..., description="PL inquiries L6M vs ever")
    num_std: int = Field(..., description="Total standard payments")
    pct_PL_enq_L6m_of_L12m: float = Field(..., description="PL inquiries L6M vs L12M")
    num_deliq_6mts: int = Field(..., description="Delinquencies in last 6 months")
    num_deliq_12mts: int = Field(..., description="Delinquencies in last 12 months")
    recent_level_of_deliq: int = Field(..., description="Recent delinquency level")
    num_std_6mts: int = Field(..., description="Standard payments in last 6 months")
    time_since_recent_enq: int = Field(..., description="Time since recent inquiry")
    num_times_60p_dpd: int = Field(..., description="Times 60+ days past due")
    num_times_30p_dpd: int = Field(..., description="Times 30+ days past due")
    max_recent_level_of_deliq: int = Field(..., description="Max recent delinquency level")

@app.get("/health", tags=["System"])
async def health_check():
    """Endpoint for load balancers to verify service health."""
    return {"status": "healthy", "service": "Credit Risk API"}

@app.post("/predict", tags=["Risk Scoring"])
async def predict_approval(data: ApplicantData):
    try:
        # 1. Convert to DataFrame
        df_input = pd.DataFrame([data.model_dump()])
        
        # 2. Ensure absolute structural parity with training data
        X_final = df_input[top_features]
        
        # 3. Predict and get probabilities
        prediction_encoded = model.predict(X_final)
        predicted_class_idx = int(prediction_encoded[0])
        confidence_scores = model.predict_proba(X_final).tolist()[0]
        
        # 4. Decode the prediction
        prediction_label = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        class_probabilities = {
            label_encoder.inverse_transform([i])[0]: round(score, 4) 
            for i, score in enumerate(confidence_scores)
        }

        # 5. Calculate SHAP values for Explainability
        # Note: For multi-class, SHAP returns a list of arrays (one for each class).
        # We extract the explanation for the specific class that was predicted.
        # 5. Calculate SHAP values for Explainability
        shap_values = explainer.shap_values(X_final)
        
        # --- THE FIX: Handle SHAP 3D Array formats properly ---
        if isinstance(shap_values, list):
            # Older SHAP behavior
            class_shap_values = shap_values[predicted_class_idx][0]
            base_val = explainer.expected_value[predicted_class_idx]
        else:
            # Newer SHAP behavior: 3D array (samples, features, classes)
            class_shap_values = shap_values[0, :, predicted_class_idx]
            base_val = explainer.expected_value[predicted_class_idx]
            
        # Create a dictionary mapping feature names to their SHAP contribution
        explanation = {
            feature: round(float(value), 4) 
            for feature, value in zip(top_features, class_shap_values)
        }

        # Sort explanation by absolute impact (highest impact first)
        sorted_explanation = dict(sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True))

        return {
            "status": "success",
            "predicted_tier": prediction_label,
            "confidence_scores": class_probabilities,
            "explanation": {
                "base_value": round(float(base_val), 4),
                "feature_contributions": sorted_explanation
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)