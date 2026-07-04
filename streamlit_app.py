import streamlit as pd_st
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Page Configurations
st.set_page_config(page_title="Thermal Power Station Dashboard", layout="wide")

st.title("🏭 ML-Based Condition Monitoring & Performance Analysis Dashboard")
st.markdown("### Project Module: Boiler Efficiency Monitoring System")
st.write("---")

# =====================================================================
# DATA & MODEL TRAINING (Cached to run only once)
# =====================================================================
@st.cache_data
def load_and_train():
    df = pd.read_csv("industrial_dataset.csv")
    df.columns = df.columns.str.strip()
    
    TARGET_COL = 'Boiler Eff (%)'
    numeric_df = df.select_dtypes(include=[np.number])
    clean_df = numeric_df.fillna(numeric_df.median())
    
    X = clean_df.drop(columns=[TARGET_COL])
    y = clean_df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test, X

model, X_test, y_test, X_features = load_and_train()
y_pred = model.predict(X_test)

# Sidebar Configuration Controls
st.sidebar.header("🎛️ Control Panel")
EFFICIENCY_THRESHOLD = st.sidebar.slider(
    "Set Alert Threshold (%)", 
    min_value=75.0, max_value=95.0, value=85.0, step=0.5
)

# Tabs matching your project structure
tab1, tab2, tab3 = st.tabs([
    "📈 Module 1: Live Condition Monitoring", 
    "📊 Module 2: Performance Analysis", 
    "🎯 Module 3: Model Verification"
])

# =====================================================================
# TAB 1: REAL-TIME MONITORING SIMULATION
# =====================================================================
with tab1:
    st.subheader("📡 Live Stream Simulation (Test Samples)")
    
    # Setup continuous dashboard layout metrics
    col1, col2, col3 = st.columns(3)
    metric_efficiency = col1.empty()
    metric_status = col2.empty()
    metric_samples = col3.empty()
    
    chart_placeholder = st.empty()
    alert_placeholder = st.empty()
    
    start_sim = st.button("▶️ Start Live Telemetry Simulation")
    
    if start_sim:
        history = []
        alert_logs = []
        
        # Loop through testing data points to simulate live telemetry
        for i in range(len(X_test)):
            sample = X_test.iloc[[i]]
            actual_eff = y_test.iloc[i]
            pred_eff = model.predict(sample)[0]
            
            history.append(pred_eff)
            if len(history) > 30: # Limit graph to last 30 intervals
                history.pop(0)
                
            # Update Dashboard Metrics
            metric_efficiency.metric("Predicted Boiler Efficiency", f"{pred_eff:.2f} %")
            
            if pred_eff < EFFICIENCY_THRESHOLD:
                metric_status.error("CRITICAL: Low Efficiency")
                alert_logs.append({
                    "Timestamp": f"Incident Index #{i}",
                    "Predicted Efficiency (%)": round(pred_eff, 2),
                    "Target Status": "Action Required"
                })
            else:
                metric_status.success("STATUS: Normal Operations")
                
            metric_samples.metric("Monitored Points Count", f"{i + 1}")
            
            # Update Live Trend Chart
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(history, marker='o', color='dodgerblue', label="Predicted Efficiency Line")
            ax.axhline(EFFICIENCY_THRESHOLD, color='red', linestyle='--', label="Alert Limit Line")
            ax.set_ylim(min(y_test) - 2, max(y_test) + 2)
            ax.set_title("Real-Time Prediction Flow")
            ax.legend(loc="lower left")
            chart_placeholder.pyplot(fig)
            plt.close(fig)
            
            # Print Alert Tables Dynamically 
            if alert_logs:
                with alert_placeholder.container():
                    st.warning("⚠️ active maintenance alerts logged:")
                    st.dataframe(pd.DataFrame(alert_logs).tail(5), use_container_width=True)
                    
            time.sleep(0.4) # Control speed of stream simulator

# =====================================================================
# TAB 2: PERFORMANCE ANALYSIS (Feature Importances)
# =====================================================================
with tab2:
    st.subheader("🛠️ Component Parameters Impact Analysis")
    st.write("Identifies which physical constraints most directly degrade or optimize boiler efficiency.")
    
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Parameter': X_features.drop(columns=['Boiler Eff (%)']).columns, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Importance', y='Parameter', data=importance_df, palette="Blues_r", ax=ax2)
    ax2.set_title("Top 10 Operational Parameters Driving Boiler Efficiency Rankings")
    st.pyplot(fig2)
    plt.close(fig2)

# =====================================================================
# TAB 3: MODEL PERFORMANCE VERIFICATION
# =====================================================================
with tab3:
    st.subheader("🎯 Model Fit Verification Statistics")
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    c1, c2 = st.columns(2)
    c1.metric("Model Confidence (R² Score)", f"{r2*100:.2f} %")
    c2.metric("Prediction Margin Error (RMSE)", f"{rmse:.3f} %")
    
    st.write("---")
    st.write("**Actual vs. Predicted Intersection Graph**")
    
    # Replaces traditional Classification confusion matrix with accurate Regression scatter matrix
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.scatter(y_test, y_pred, alpha=0.5, color='dodgerblue', edgecolors='k')
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Identity / Perfect Accuracy Line")
    ax3.set_xlabel("Actual Ground Truth Values (%)")
    ax3.set_ylabel("Machine Learning Output Predictions (%)")
    ax3.legend()
    
    st.pyplot(fig3)
    plt.close(fig3)
