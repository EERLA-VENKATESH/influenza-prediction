import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance

# Cache data loading and preprocessing
@st.cache_data
def load_and_preprocess(uploaded_file):
    df = pd.read_csv(uploaded_file, skiprows=1)
    df.replace('X', pd.NA, inplace=True)
    
    numeric_columns = ['% WEIGHTED ILI', '%UNWEIGHTED ILI', 'AGE 0-4', 
                      'AGE 25-49', 'AGE 25-64', 'AGE 5-24', 'AGE 50-64', 
                      'AGE 65', 'ILITOTAL', 'NUM. OF PROVIDERS', 'TOTAL PATIENTS']
    
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + df['WEEK'].astype(str) + '1', 
                               format='%Y%W%w')
    df.set_index('DATE', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

st.title("🦠 Advanced ILI Forecasting Dashboard")
st.markdown("### Multi-Model Comparison with Temporal Validation")

with st.expander("ℹ️ How to use this app"):
    st.write("""
    1. Upload your ILI surveillance data (CSV format)
    2. Select target variable and features
    3. Adjust model parameters
    4. View performance metrics and forecasts
    """)

uploaded_file = st.file_uploader("Upload CDC-style ILI data", type=["csv"], 
                                help="Expected format: CDC CSV structure with weekly reports")

if uploaded_file:
    df = load_and_preprocess(uploaded_file)
    
    # Data preview section
    with st.expander("🔍 Data Preview"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**First 5 rows:**")
            st.dataframe(df.head())
        with col2:
            st.write("**Summary Statistics:**")
            st.write(df.describe())

    # Feature Selection
    st.sidebar.header("⚙️ Model Configuration")
    target = st.sidebar.selectbox("Target Variable", df.select_dtypes(include=np.number).columns)
    available_features = [col for col in df.columns if col != target]
    selected_features = st.sidebar.multiselect("Features", available_features, 
                                              default=['AGE 0-4', 'TOTAL PATIENTS'],
                                              help="Select epidemiological features")

    # Model parameters
    model_choice = st.sidebar.radio("Models", ["Random Forest", "Decision Tree"], 
                                   help="Compare different tree-based approaches")
    
    n_splits = st.sidebar.slider("Time Series Splits", 3, 10, 5,
                                help="Number of temporal cross-validation splits")
    
    # Enhanced model configuration
    if model_choice == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 100)
        max_depth = st.sidebar.slider("Max Depth", 2, 20, 5)
    else:
        max_depth = st.sidebar.slider("Max Depth", 2, 20, 5)

    # Temporal cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X = df[selected_features]
    y = df[target]

    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Model training with caching
    @st.cache_resource
    def train_model(_model, X_train, y_train):
        return _model.fit(X_train, y_train)

    metrics = []
    feature_importances = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        with st.spinner(f"Training fold {fold+1}/{n_splits}..."):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if model_choice == "Random Forest":
                model = RandomForestRegressor(n_estimators=n_estimators, 
                                            max_depth=max_depth,
                                            random_state=42)
            else:
                model = DecisionTreeRegressor(max_depth=max_depth,
                                            random_state=42)
            
            model = train_model(model, X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Store metrics
            fold_metrics = {
                'fold': fold+1,
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R²': r2_score(y_test, y_pred)
            }
            metrics.append(fold_metrics)
            
            # Feature importance analysis
            if model_choice == "Random Forest":
                importances = model.feature_importances_
            else:
                result = permutation_importance(model, X_test, y_test, n_repeats=10)
                importances = result.importances_mean
                
            feature_importances.append(pd.Series(importances, index=selected_features))
            
            progress_bar.progress((fold+1)/n_splits)
    
    st.toast("✅ Training completed!", icon="✅")
    
    # Performance visualization
    st.header("📊 Model Evaluation")
    metrics_df = pd.DataFrame(metrics).set_index('fold')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average MAE", f"{metrics_df.MAE.mean():.2f} ± {metrics_df.MAE.std():.2f}")
    with col2:
        st.metric("Average RMSE", f"{metrics_df.RMSE.mean():.2f} ± {metrics_df.RMSE.std():.2f}")
    with col3:
        st.metric("Average R²", f"{metrics_df['R²'].mean():.2f} ± {metrics_df['R²'].std():.2f}")
    
    # Feature importance visualization
    st.subheader("🔍 Feature Importance Analysis")
    importance_df = pd.concat(feature_importances, axis=1).mean(axis=1).sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    importance_df.plot(kind='barh', ax=ax)
    ax.set_title("Aggregated Feature Importances")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)
    
    # Forecast visualization
    st.subheader("📈 Temporal Validation Results")
    fig, ax = plt.subplots(figsize=(12, 6))
    df[target].plot(ax=ax, label='Actual', color='blue')
    
    # Generate predictions for visualization
    full_pred = model.predict(X)
    ax.plot(df.index, full_pred, label='Predicted', linestyle='--', color='red')
    
    ax.set_title(f"{model_choice} Forecast vs Actuals")
    ax.legend()
    st.pyplot(fig)
    
    # Model persistence
    st.download_button("💾 Download Trained Model",
                      data=joblib.dump(model, 'model.joblib')[0],
                      file_name="ili_forecast_model.joblib",
                      mime="application/octet-stream")

# Add footer
st.markdown("---")
st.caption("Built with Streamlit | Best practices from epidemiological forecasting research [1][3][6]")
