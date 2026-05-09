<h1 align="center">💳 Credit Risk Prediction Microservice</h1>

<p align="center">
  <b>Production-ready ML system for credit risk classification with explainability</b><br>
  Built using XGBoost, FastAPI, SHAP, and Optuna
</p>

<hr>

<h2>🚀 Overview</h2>
<p>
This project is a <b>4-tier credit risk classification system</b> designed to predict the creditworthiness of applicants using machine learning. 
It processes real-world financial data and outputs both <b>risk predictions</b> and <b>interpretable explanations</b> for each decision.
</p>

<ul>
  <li>Handles <b>42,000+ applicant records</b></li>
  <li>Addresses <b>class imbalance (60% majority class)</b></li>
  <li>Provides <b>real-time predictions via API</b></li>
  <li>Ensures <b>model transparency using SHAP</b></li>
</ul>

<hr>

<h2>🧠 Model Details</h2>

<ul>
  <li><b>Algorithm:</b> XGBoost (Multi-class classification)</li>
  <li><b>Classes:</b> 4 Risk Levels (P1, P2, P3, P4)</li>
  <li><b>Evaluation Metric:</b> Macro F1 Score</li>
  <li><b>Final Score:</b> <b>0.75 Macro F1</b></li>
</ul>

<hr>

<h2>📊 Feature Engineering</h2>

<p>
A robust statistical pipeline was designed to reduce noise and improve model generalization:
</p>

<ul>
  <li>Chi-Square Test → Categorical feature relevance</li>
  <li>ANOVA → Numerical feature importance</li>
  <li>VIF (Variance Inflation Factor) → Multicollinearity removal</li>
</ul>

<p>
<b>Result:</b> Reduced <b>80+ features → 15 high-signal features</b> while eliminating data leakage.
</p>

<hr>

<h2>⚙️ Hyperparameter Optimization</h2>

<ul>
  <li>Framework: Optuna</li>
  <li>Trials: 50+</li>
  <li>Hardware: T4 GPU</li>
</ul>

<p>
Model selection prioritized <b>correct detection of high-risk applicants</b> over raw accuracy,
ensuring real-world reliability.
</p>

<hr>

<h2>🔍 Explainability (SHAP)</h2>

<p>
Integrated <b>SHAP (SHapley Additive exPlanations)</b> to provide:
</p>

<ul>
  <li>Per-feature contribution scores</li>
  <li>Transparent decision-making</li>
  <li>Regulatory-friendly outputs</li>
</ul>

<hr>

<h2>🌐 API Deployment</h2>

<ul>
  <li>Framework: FastAPI</li>
  <li>Architecture: Microservice</li>
  <li>Response: JSON with prediction + explanation</li>
</ul>

<p><b>Sample Response:</b></p>

<pre>
{
  "risk_category": "P4",
  "confidence": 0.87,
  "top_features": {
    "income": -0.21,
    "delinquencies": 0.35,
    "credit_utilization": 0.18
  }
}
</pre>

<hr>

<h2>📁 Project Structure</h2>

<pre>
├── model/
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│
├── notebooks/
│   ├── training.ipynb
│
├── src/
│   ├── feature_engineering.py
│   ├── predict.py
│
├── api/
│   ├── main.py
│
├── requirements.txt
└── README.md
</pre>

<hr>

<h2>🛠️ Tech Stack</h2>

<ul>
  <li>Python</li>
  <li>XGBoost</li>
  <li>FastAPI</li>
  <li>Optuna</li>
  <li>SHAP</li>
  <li>Pandas / NumPy / Scikit-learn</li>
</ul>

<hr>

<h2>▶️ How to Run</h2>

<pre>
# Clone the repo
git clone https://github.com/your-username/credit-risk-microservice.git

# Navigate into the project
cd credit-risk-microservice

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn api.main:app --reload
</pre>

<p>API will be available at:</p>
<pre>http://127.0.0.1:8000</pre>

<hr>

<h2>🎯 Key Highlights</h2>

<ul>
  <li>Production-ready ML microservice</li>
  <li>Strong handling of imbalanced data</li>
  <li>Explainable AI integration (SHAP)</li>
  <li>Optimized using Optuna</li>
  <li>Designed with real-world financial constraints in mind</li>
</ul>

<hr>

<h2>📌 Future Improvements</h2>

<ul>
  <li>Add CI/CD pipeline</li>
  <li>Deploy on cloud (AWS / GCP)</li>
  <li>Add user authentication & dashboard</li>
  <li>Model monitoring (drift detection)</li>
</ul>

<hr>

<h2>👨‍💻 Author</h2>

<p>
<b>Pranav Marwaha</b><br>
Final Year Student | Aspiring Data Scientist
</p>

<p align="center">⭐ If you like this project, consider giving it a star!</p>
