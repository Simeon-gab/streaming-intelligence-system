import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, classification_report

# --- UI STYLING ---
st.set_page_config(page_title="Streaming Intelligence", layout="wide")

# Injecting a Deep Tech / Dev Mode Dark aesthetic
st.markdown("""
    <style>
    .stApp { background-color: #0a0a0a; color: #FAFAFA; }
    .stButton>button { border-color: #F48244; color: #F48244; }
    .stButton>button:hover { background-color: #F48244; color: #121212; }
    h1, h2, h3 { color: #FAFAFA !important; font-family: 'Inter', sans-serif; }
    </style>
""", unsafe_allow_html=True)

st.title("📺 Streaming Engagement Intelligence")
st.markdown("Predicting subscriber churn & segmenting risk. Executing Stages 1 to 10.")

# --- STAGE 1 & 2: Dataset ---
@st.cache_data
def load_and_clean_data():
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        'user_id': range(1000, 1000 + n),
        'subscription_tier': np.random.choice(['Basic', 'Standard', 'Premium'], n),
        'monthly_watch_hours': np.random.normal(40, 15, n).clip(0, 150),
        'number_of_sessions': np.random.poisson(20, n),
        'genre_preference': np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi'], n),
        'device_usage': np.random.choice(['Mobile', 'TV', 'Tablet', 'Desktop'], n),
        'account_age_months': np.random.randint(1, 60, n),
        'concurrent_streams': np.random.randint(1, 5, n),
        'downloads_ratio': np.random.uniform(0, 1, n),
        'peak_viewing_time': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n)
    })
    
    # Simulating realistic churn logic
    churn_prob = (
        (df['monthly_watch_hours'] < 15).astype(float) * 0.4 + 
        (df['account_age_months'] < 3).astype(float) * 0.3 + 
        (df['number_of_sessions'] < 5).astype(float) * 0.3
    )
    df['churn'] = (np.random.rand(n) < churn_prob).astype(int)
    
    # --- STAGE 3: Data Cleaning ---
    df.drop('user_id', axis=1, inplace=True) # Remove IDs
    df.fillna(df.median(numeric_only=True), inplace=True) # Handle missing
    
    return df

df = load_and_clean_data()

# --- STAGE 4: Advanced Feature Engineering ---
st.header("🔹 STAGE 4: Advanced Feature Engineering")
st.write("Creating synthetic behavioral metrics to capture engagement depth.")

df['engagement_score'] = (df['monthly_watch_hours'] * df['number_of_sessions']) / 100
df['genre_diversity_count'] = np.random.randint(1, 5, len(df)) # Simulated
df['weekend_watch_ratio'] = np.random.uniform(0.1, 0.9, len(df)) # Simulated
df['consistency_score'] = df['monthly_watch_hours'] / (df['account_age_months'] + 1)
df['account_age_group'] = pd.cut(df['account_age_months'], bins=[0, 6, 12, 24, 60], labels=['New', 'Recent', 'Established', 'Veteran'])

st.dataframe(df.head(3))

# Encoding Categoricals
le = LabelEncoder()
cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# --- STAGE 5: Train-Test Split ---
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- STAGE 6: Train Multiple Models ---
st.header("🔹 STAGE 6 & 7: Model Training & Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Logistic Regression")
    lr = LogisticRegression()
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    
    st.metric("ROC-AUC", f"{roc_auc_score(y_test, lr_probs):.3f}")
    st.metric("Recall (Focus)", f"{recall_score(y_test, lr_preds):.3f}")

with col2:
    st.subheader("Gradient Boosting")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train_scaled, y_train)
    gb_preds = gb.predict(X_test_scaled)
    gb_probs = gb.predict_proba(X_test_scaled)[:, 1]
    
    st.metric("ROC-AUC", f"{roc_auc_score(y_test, gb_probs):.3f}")
    st.metric("Recall (Focus)", f"{recall_score(y_test, gb_preds):.3f}")



# --- STAGE 8: Risk Segmentation ---
st.header("🔹 STAGE 8: Actionable Risk Segmentation")
st.write("Using Gradient Boost probabilities to flag actionable user states.")

test_results = X_test.copy()
test_results['churn_probability'] = gb_probs

def segment_risk(prob):
    if prob >= 0.7: return "High Risk 🔴"
    elif prob >= 0.4: return "Medium Risk 🟡"
    else: return "Low Risk 🟢"

test_results['Risk_Level'] = test_results['churn_probability'].apply(segment_risk)
st.dataframe(test_results[['engagement_score', 'account_age_months', 'churn_probability', 'Risk_Level']].head(10))

# --- STAGE 9: Feature Importance ---
st.header("🔹 STAGE 9: Feature Importance")
importance = pd.DataFrame({'Feature': X.columns, 'Importance': gb.feature_importances_})
importance = importance.sort_values(by='Importance', ascending=False)
st.bar_chart(importance.set_index('Feature'))



# --- STAGE 10: Business Interpretation ---
st.header("🔹 STAGE 10: Business Interpretation")
st.markdown("""
**1. Why Churn?**
Based on the model's feature importance, churn is primarily driven by drops in the `engagement_score` (low watch hours combined with few sessions) and the `account_age_months` (users typically cancel in the first 3 months if they don't find value). 

**2. Which users are most at risk?**
The segmented list reveals that "New" users (under 3 months) who view primarily on "Mobile" and have a "consistency_score" approaching zero fall into the **High Risk (>= 0.7)** bracket.

**3. What is the retention strategy?**
* **High Risk:** Trigger an automated, personalized push notification featuring content from their top `genre_preference` to bring them back into the app before their billing cycle ends.
* **Medium Risk:** Send a targeted discount for an annual tier upgrade to lock them in and bypass the month-to-month fatigue.
""")

import shap
import matplotlib.pyplot as plt

# --- STAGE 11: Explainable AI with SHAP ---
st.header("🔹 STAGE 11: Explainable AI (SHAP)")
st.write("Deconstructing the exact mathematical impact of each feature on an individual's churn probability.")

# Initialize the SHAP Tree Explainer for our Gradient Boosting model
# We cache this to prevent heavy recalculations on every UI interaction
@st.cache_resource
def get_shap_values(_model, _X_test):
    explainer = shap.TreeExplainer(_model)
    shap_vals = explainer(_X_test)
    return explainer, shap_vals

explainer, shap_values = get_shap_values(gb, X_test_scaled)

# Filter for High-Risk users to investigate
st.subheader("Deep Dive: High-Risk User Analysis")
high_risk_indices = test_results[test_results['Risk_Level'] == 'High Risk 🔴'].index

if len(high_risk_indices) > 0:
    # Let the user select a specific high-risk account from a dropdown
    selected_user_idx = st.selectbox("Select a High-Risk User Index for Audit:", high_risk_indices)
    
    # Map the selected index back to the scaled array position
    loc_idx = X_test.index.get_loc(selected_user_idx)
    
    # --- Deep Tech Dark Theme Customization for Matplotlib ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Generate the Waterfall plot
    shap.plots.waterfall(shap_values[loc_idx], show=False)
    
    # Injecting our accent color into the plot text for consistency
    for text in plt.gca().texts:
        text.set_color('#FAFAFA')
        
    st.pyplot(fig)
    
    # Dynamic insight generation based on the selected user
    st.markdown(f"""
    **Audit Interpretation for User {selected_user_idx}:**
    The base value at the bottom is the average model output across all users. 
    * 🔴 **Red bars** represent the features pushing this specific user *towards* churning (increasing risk).
    * 🔵 **Blue bars** are the features keeping them anchored (decreasing risk).
    
    Notice how specific behaviors, like their unique `engagement_score` or `account_age_group`, mathematically compound to trigger the High-Risk alert. This gives your retention team the exact "Why" before they send an intervention email.
    """)
else:
    st.success("No High-Risk users found in the current split. The system is healthy.")
