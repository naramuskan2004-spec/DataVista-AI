import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio

# ML Libraries
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Theme
pio.templates.default = "plotly_white"

# ----------------------------
# APP CONFIG
# ----------------------------
st.set_page_config(
    page_title="DataVista AI",
    page_icon="✨",
    layout="wide"
)

# ----------------------------
# HEADER
# ----------------------------
st.title("✨ DataVista AI: Smart Data Visualization Dashboard")
st.markdown("##### By Arshpreet | MSc Data Science")
st.divider()

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# ----------------------------
# MAIN APP
# ----------------------------
if uploaded_file is not None:

    # Load Data
    df = pd.read_csv(uploaded_file)

    # Clean columns
    df.columns = df.columns.str.strip().str.lower()

    # Remove outliers (top 1%)
    if 'price' in df.columns:
        df = df[df['price'] < df['price'].quantile(0.99)]

    # Numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # ----------------------------
    # DATA OVERVIEW
    # ----------------------------
    st.header("📊 Data Overview")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Rows", len(df))
        st.metric("Columns", len(df.columns))
        st.info(f"Numeric Columns: {len(numeric_cols)}")

    with col2:
        st.dataframe(df.head(), use_container_width=True)

    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe().T, use_container_width=True)

    st.divider()

    # ----------------------------
    # VISUALIZATION
    # ----------------------------
    st.header("📊 Data Visualizations")

    if len(numeric_cols) >= 2:

        st.subheader("🔵 Scatter Plot")
        x_axis = st.selectbox("X-axis", numeric_cols)
        y_axis = st.selectbox("Y-axis", numeric_cols, index=1)

        fig_scatter = px.scatter(df, x=x_axis, y=y_axis)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("📈 Line Chart")
        line_col = st.selectbox("Line Column", numeric_cols)

        sorted_df = df.sort_values(by=line_col)
        fig_line = px.line(sorted_df, y=line_col)
        st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("📊 Histogram")
        hist_col = st.selectbox("Histogram Column", numeric_cols)

        fig_hist = px.histogram(df, x=hist_col)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("📊 Bar Chart")
        bar_col = st.selectbox("Bar Column", numeric_cols)

        top_df = df.sort_values(by=bar_col, ascending=False).head(20)
        fig_bar = px.bar(top_df, y=bar_col)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("📦 Box Plot")
        box_col = st.selectbox("Box Column", numeric_cols)

        fig_box = px.box(df, y=box_col)
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("🔥 Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        fig_heatmap = px.imshow(corr, text_auto=True, color_continuous_scale="Blues")
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.divider()

    # ----------------------------
    # MACHINE LEARNING
    # ----------------------------
    st.header("🤖 Machine Learning Prediction")

    target_col = st.selectbox("Select Target Column", numeric_cols)

    available_features = [col for col in numeric_cols if col != target_col]
    feature_cols = st.multiselect("Select Feature Columns", available_features)

    model_choice = st.selectbox("Choose Model", ["Linear Regression", "Decision Tree"])

    if len(feature_cols) > 0:

        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if model_choice == "Linear Regression":
            model = LinearRegression()
        else:
            model = DecisionTreeRegressor(max_depth=5)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)

        st.info(f"Model Accuracy (R² Score): {round(score, 2)}")

        st.subheader("Enter Values for Prediction")

        input_data = []
        for col in feature_cols:
            val = st.number_input(f"{col}", value=float(df[col].mean()))
            input_data.append(val)

        if st.button("Predict"):
            prediction = model.predict([input_data])
            st.success(f"Predicted {target_col}: {round(prediction[0], 2)}")

    else:
        st.warning("Please select feature columns")

    st.divider()

    # ----------------------------
    # SMART AI ASSISTANT
    # ----------------------------
    # ----------------------------

# ----------------------------
st.header("🤖 AI Data Assistant")

query = st.text_input("Ask anything about your dataset...")

if query:
    q = query.lower()

    st.subheader("💡 Insight")

    # Average
    if "average" in q or "mean" in q:
        st.write("Here are the average values:")
        for col in numeric_cols:
            st.write(f"• {col}: {round(df[col].mean(), 2)}")

    # Max
    elif "max" in q or "highest" in q:
        st.write("Maximum values in dataset:")
        for col in numeric_cols:
            st.write(f"• {col}: {df[col].max()}")

    # Min
    elif "min" in q or "lowest" in q:
        st.write("Minimum values in dataset:")
        for col in numeric_cols:
            st.write(f"• {col}: {df[col].min()}")

    # Correlation
    elif "correlation" in q or "relationship" in q:
        corr = df.corr(numeric_only=True)
        st.write("Here is the correlation between variables:")
        st.dataframe(corr)

        st.write("Strong relationships:")
        st.write(corr.unstack().sort_values(ascending=False).drop_duplicates().head(5))

    # Important features
    elif "important" in q or "impact" in q or "affect" in q or "influence" in q:
        if 'price' in df.columns:
            corr = df.corr(numeric_only=True)
            st.write("Features impacting price:")
            st.write(corr['price'].sort_values(ascending=False))

    # Insights
    elif "insight" in q or "analysis" in q:
        st.write("Key Insights:")
        st.write(f"- Average price: {round(df['price'].mean(),2)}")
        st.write(f"- Max price: {df['price'].max()}")
        st.write(f"- Strongest factor: sqft_living")

    # Summary
    elif "summary" in q or "info" in q:
        st.write("Dataset Overview:")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")
        st.write(list(df.columns))

    # Charts
    elif "chart" in q:
        st.write("Suggested charts:")
        st.write("- Scatter → relationships")
        st.write("- Histogram → distribution")
        st.write("- Bar → comparison")

    # Prediction
    elif "predict" in q:
        st.write("Go to ML section to predict house prices using selected features.")

    else:
        st.write("Try asking:")
        st.write("- What is average price?")
        st.write("- Show correlation")
        st.write("- Give insights")

else:
    st.info("👈 Upload a CSV file to begin")