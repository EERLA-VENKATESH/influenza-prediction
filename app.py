import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from io import BytesIO

st.set_page_config(page_title="ILI Forecasting", layout="wide")

# Custom CSS Styling
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
            color: #1a1a1a;
        }
        h1, h2, h3 {
            color: #0077b6;
        }
        .stButton>button {
            background-color: #0077b6;
            color: white;
        }
        .stDownloadButton>button {
            background-color: #48cae4;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🦠 Influenza-Like Illness (ILI) Forecasting using ML Models")

# File Upload
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file:
    st.sidebar.markdown("## ⚙️ Model & Data Settings")
    df = pd.read_csv(uploaded_file, skiprows=1)
    df.replace('X', pd.NA, inplace=True)

    # Convert numeric
    numeric_columns = ['% WEIGHTED ILI', '%UNWEIGHTED ILI', 'AGE 0-4', 'AGE 25-49',
                       'AGE 25-64', 'AGE 5-24', 'AGE 50-64', 'AGE 65', 'ILITOTAL',
                       'NUM. OF PROVIDERS', 'TOTAL PATIENTS']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Date conversion
    df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + df['WEEK'].astype(str) + '1', format='%Y%W%w')
    df.set_index('DATE', inplace=True)
    df.fillna(method='ffill', inplace=True)

    # Preview & Summary
    with st.expander("🔍 Preview & Summary"):
        st.dataframe(df.head(10))
        st.dataframe(df.describe())

    # Date Range Filtering
    min_date, max_date = df.index.min(), df.index.max()
    date_range = st.sidebar.date_input("📅 Filter Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    if len(date_range) == 2:
        df = df.loc[date_range[0]:date_range[1]]

    # Feature selection
    target = st.sidebar.selectbox("🎯 Target Variable", numeric_columns)
    selected_features = st.sidebar.multiselect("📌 Input Features", df.columns, default=['YEAR', 'AGE 0-4', 'AGE 25-49'])
    if target in selected_features:
        selected_features.remove(target)

    X = df[selected_features]
    y = df[target]

    # Optional Normalization
    normalize = st.sidebar.checkbox("🔄 Normalize Input Features")
    if normalize:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=selected_features)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Model
    model_choice = st.sidebar.radio("🧠 Choose Model", ["Random Forest", "Decision Tree"])
    model = RandomForestRegressor(n_estimators=100, random_state=42) if model_choice == "Random Forest" else DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Model Performance")
    st.success(f"**MAE:** {mae:.4f}")
    st.success(f"**RMSE:** {rmse:.4f}")
    st.success(f"**R² Score:** {r2:.4f}")

    # Plotly Chart
    st.subheader("📉 Actual vs Predicted (Interactive)")
    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual', line=dict(color='blue')))
    fig_plotly.add_trace(go.Scatter(x=y_test.index, y=y_pred, mode='lines', name='Predicted', line=dict(color='red', dash='dash')))
    fig_plotly.update_layout(title=f"{model_choice}: {target} Forecast", xaxis_title='Date', yaxis_title=target)
    st.plotly_chart(fig_plotly, use_container_width=True)

    # Feature importance
    if model_choice == "Random Forest":
        st.subheader("📊 Feature Importance")
        importances = pd.Series(model.feature_importances_, index=selected_features)
        sorted_imp = importances.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sorted_imp.plot(kind='barh', color='green', ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

    # Correlation heatmap
    st.subheader("🧪 Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[numeric_columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    # Download option
    st.subheader("💾 Download Forecasts")
    output_df = pd.DataFrame({
        "Date": y_test.index,
        "Actual": y_test.values,
        "Predicted": y_pred
    })
    buffer = BytesIO()
    output_df.to_csv(buffer, index=False)
    st.download_button("📥 Download CSV", buffer.getvalue(), "ILI_Forecast.csv", mime="text/csv")

    # Sidebar info
    with st.sidebar.expander("🧠 About This App"):
        st.markdown("""
        - **Goal**: Forecast Influenza-Like Illness cases.
        - **Models**: Random Forest & Decision Tree.
        - **Input**: Weekly records of age groups, ILI data, provider counts.
        - **Output**: Forecast + Evaluation + Graphs.
        """)

    # FAQ
    with st.expander("❓ FAQ / About Model"):
        st.markdown("""
        **Q: What is ILI?**  
        A: Influenza-Like Illness is a syndrome defined by symptoms such as fever, cough, and sore throat.

        **Q: Which model is better?**  
        A: Random Forest usually performs better in generalizing for time-based health data.

        **Q: Can I add more models?**  
        A: Yes! You can extend the code with XGBoost, SVR, or LSTM models.
        """)
