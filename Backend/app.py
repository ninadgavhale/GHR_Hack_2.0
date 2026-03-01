# app.py
# Explainable & Fair AI Loan Decision Backend (Flask)

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)

# ✅ Proper CORS Configuration (adjust port if needed)
CORS(
    app,
    resources={r"/*": {"origins": ["http://localhost:3000"]}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    supports_credentials=False
)
#   MODEL & DATA (loaded once at startup)

np.random.seed(42)

# Generate synthetic training data (realistic-ish credit data)
n_samples = 18000
data = pd.DataFrame({
    'age': np.random.randint(21, 68, n_samples),
    'annual_income': np.round(np.random.normal(62000, 22000, n_samples)).clip(18000, 280000).astype(int),
    'employment_years': np.random.randint(0, 42, n_samples),
    'credit_score': np.random.randint(320, 850, n_samples),
    'loan_amount': np.random.randint(8000, 450000, n_samples),
    'debt_to_income': np.round(np.random.uniform(0.04, 0.58, n_samples), 3),
    'education': np.random.choice(['High School', 'Graduate', 'Postgraduate'], n_samples, p=[0.32, 0.48, 0.20]),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.42, 0.48, 0.10]),
    'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.51, 0.49]),
})

# Simulated approval probability (with small gender effect ~12–15% disparity)
logit = (
    0.000045 * data['annual_income'] +
    0.0092 * data['credit_score'] +
    0.145 * data['employment_years'] -
    9.8 * data['debt_to_income'] -
    0.000004 * data['loan_amount'] +
    0.035 * data['age'] -
    7.2 +
    np.where(data['gender'] == 'Female', -1.35, 0) +   # protected attribute effect
    np.random.normal(0, 1.4, n_samples)
)

proba = 1 / (1 + np.exp(-logit))
data['approved'] = (np.random.rand(n_samples) < proba).astype(int)

# Precompute group approval rates (for bias report)
male_rate   = data[data['gender']=='Male']['approved'].mean()   * 100
female_rate = data[data['gender']=='Female']['approved'].mean() * 100
bias_disparity = round(abs(male_rate - female_rate), 1)

# Prepare features for modeling
num_features = ['age', 'annual_income', 'employment_years', 'credit_score', 'loan_amount', 'debt_to_income']
cat_features = ['education', 'marital_status', 'gender']

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
X_cat = encoder.fit_transform(data[cat_features])
X_num = data[num_features].values

X = np.hstack([X_num, X_cat])
y = data['approved']

# Train model
model = RandomForestClassifier(
    n_estimators=180,
    max_depth=11,
    min_samples_leaf=8,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

# SHAP explainer 
explainer = shap.TreeExplainer(model)

print(f"Model loaded — overall approval rate: {y.mean():.1%}")
print(f"Gender approval rates → Male: {male_rate:.1f}%, Female: {female_rate:.1f}%, Disparity: {bias_disparity:.1f}%")


#   HELPER — map feature names back to readable text

def human_readable_feature(feat):
    if feat.startswith('education_'): return f"{feat.split('_')[-1]} Education"
    if feat.startswith('marital_status_'): return f"{feat.split('_')[-1]}"
    if feat == 'annual_income':   return "High Income"
    if feat == 'credit_score':    return "Good Credit Score"
    if feat == 'employment_years': return "Strong Employment History"
    if feat == 'debt_to_income':  return "Low Debt Ratio"
    if feat == 'loan_amount':     return "Loan Amount"
    if feat == 'age':             return "Age"
    return feat.replace('_', ' ').title()


#   API ENDPOINT

@app.route('/evaluate', methods=['POST'])
def evaluate_loan():
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "No JSON payload"}), 400

        # Extract fields (all required — frontend should validate)
        inp = {
            'age':               float(payload.get('age', 0)),
            'annual_income':     float(payload.get('annual_income', 0)),
            'employment_years':  float(payload.get('employment_years', 0)),
            'credit_score':      float(payload.get('credit_score', 0)),
            'loan_amount':       float(payload.get('loan_amount', 0)),
            'debt_to_income':    float(payload.get('debt_to_income', 0)),
            'education':         payload.get('education', 'Graduate'),
            'marital_status':    payload.get('marital_status', 'Married'),
            'gender':            payload.get('gender', 'Male'),
        }

        # Prepare model input
        input_num = np.array([[
            inp['age'], inp['annual_income'], inp['employment_years'],
            inp['credit_score'], inp['loan_amount'], inp['debt_to_income']
        ]])

        input_cat_df = pd.DataFrame([[
            inp['education'], inp['marital_status'], inp['gender']
        ]], columns=cat_features)

        input_cat_ohe = encoder.transform(input_cat_df)
        input_X = np.hstack([input_num, input_cat_ohe])

        # Predict
        proba = model.predict_proba(input_X)[0][1]
        decision = "Approved" if proba >= 0.52 else "Rejected"   # adjustable threshold
        confidence = round(proba * 100)
        risk_level = "Low" if confidence >= 78 else "Medium" if confidence >= 55 else "High"

        # SHAP values
        shap_values = explainer.shap_values(input_X)
        # For binary classification → we usually look at the positive class (index 1)
        shap_values = explainer.shap_values(input_X)

        if isinstance(shap_values, list):
            contrib = shap_values[1][0]   # positive class
            base_value = explainer.expected_value[1]
        else:
            contrib = shap_values[0]      # single output
            base_value = explainer.expected_value
            
        contrib = np.array(contrib).flatten()


        # Create feature → contribution mapping
        feature_names = num_features + list(encoder.get_feature_names_out())
       # Ensure correct shape
        contrib = np.array(contrib)

        if len(contrib.shape) > 1:
              contrib = contrib[0]

        shap_dict = {}
        for f, v in zip(feature_names, contrib):
            shap_dict[f] = round(float(v), 4)

        # ── Build human-friendly explanation ────────────────────────────────
        pos_factors = []
        neg_factors = []

        for feat, val in sorted(shap_dict.items(), key=lambda x: x[1], reverse=True):
            if abs(val) < 0.035: continue                     # ignore tiny contributions
            if 'gender_' in feat: continue                    # hide protected attribute

            label = human_readable_feature(feat)

            if val > 0:
                pos_factors.append(f"+ {label}")
            else:
                neg_factors.append(f"- {label}")

            if len(pos_factors) + len(neg_factors) >= 6:
                break

        # ── Bias info (static for now — computed on training data) ──────────
        bias_info = {
            "male_approval_rate": round(male_rate),
            "female_approval_rate": round(female_rate),
            "disparity": bias_disparity,
            "bias_detected": bool(bias_disparity > 10),
            "action": "Review Model" if bias_disparity > 10 else "No action needed"
        }

        # Optional: SHAP plot as base64 (can be sent to frontend)
        # fig = plt.figure(figsize=(8, 5))
        # shap.plots.waterfall(
        #     shap.Explanation(
        #         values=contrib,
        #         base_values=base_value,
        #         data=input_X[0],
        #         feature_names=feature_names
        #     ),
        #     max_display=10,
        #     show=False
        # )
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png', bbox_inches='tight')
        # buf.seek(0)
        # img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        # plt.close(fig)

        # ── Final response ──────────────────────────────────────────────────
        return jsonify({
            "status": "success",
            "decision": decision,
            "confidence": confidence,
            "risk_level": risk_level,
            "positive_factors": pos_factors[:4],
            "negative_factors": neg_factors[:4],
            "bias_analysis": bias_info,
            # "shap_plot_base64": img_base64,           # optional — frontend can display
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)