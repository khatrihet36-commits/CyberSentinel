import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CyberSentinel",
    page_icon="🛡️",
    layout="wide"
)


# ==========================================
# LOAD TRAINED MODELS
# ==========================================

model = joblib.load(
    "models/cybersentinel_model_compressed.pkl"
)


category_model = joblib.load(
    "models/cybersentinel_category_model_final.pkl"
)



# ==========================================
# HEADER
# ==========================================

st.title("🛡️ CyberSentinel")

st.subheader(
    "AI-Powered Cybersecurity Threat Detection"
)

st.write(
    "CyberSentinel analyzes network traffic "
    "using Machine Learning to detect and "
    "classify potential cyber attacks."
)

st.success(
    "🤖 CyberSentinel AI models loaded successfully!"
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🛡️ CyberSentinel")

st.sidebar.info(
    """
    CyberSentinel detects:

    🟢 Normal Traffic

    🔴 Cyber Attacks

    🎯 Attack Categories
    """
)

st.sidebar.write("Model: Random Forest")
st.sidebar.write("Dataset: UNSW-NB15")


# ==========================================
# FILE UPLOAD
# ==========================================

st.header("📁 Upload Network Traffic")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


# ==========================================
# PROCESS FILE
# ==========================================

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success(
        "✅ Dataset uploaded successfully!"
    )


    # ======================================
    # DATASET INFORMATION
    # ======================================

    st.header("📊 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Records",
            len(data)
        )

    with col2:
        st.metric(
            "Total Columns",
            len(data.columns)
        )

    with col3:
        st.metric(
            "Missing Values",
            int(data.isnull().sum().sum())
        )


    # ======================================
    # DATA PREVIEW
    # ======================================

    st.subheader("👀 Dataset Preview")

    st.dataframe(
        data.head(),
        use_container_width=True
    )


    # ======================================
    # PREPARE DATA
    # ======================================

    prediction_data = data.copy()


    # Remove ID
    if "id" in prediction_data.columns:

        prediction_data = prediction_data.drop(
            columns=["id"]
        )


    # Remove target columns
    columns_to_remove = []

    if "label" in prediction_data.columns:

        columns_to_remove.append("label")

    if "attack_cat" in prediction_data.columns:

        columns_to_remove.append("attack_cat")


    if columns_to_remove:

        prediction_data = prediction_data.drop(
            columns=columns_to_remove
        )


    # ======================================
    # ENCODE DATA
    # ======================================

    prediction_data = pd.get_dummies(
        prediction_data
    )


    # ======================================
    # MATCH BINARY MODEL FEATURES
    # ======================================

    if hasattr(model, "feature_names_in_"):

        prediction_data = prediction_data.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )


    # ======================================
    # BINARY PREDICTION
    # ======================================

    with st.spinner(
        "🤖 Analyzing network traffic..."
    ):

        predictions = model.predict(
            prediction_data
        )


    # ======================================
    # CATEGORY PREDICTION
    # ======================================

    category_data = prediction_data.copy()


    if hasattr(
        category_model,
        "feature_names_in_"
    ):

        category_data = category_data.reindex(
            columns=category_model.feature_names_in_,
            fill_value=0
        )


    category_predictions = (
        category_model.predict(
            category_data
        )
    )


    # ======================================
    # ADD RESULTS
    # ======================================

    data["Prediction"] = predictions

    data["Threat"] = data["Prediction"].map({
        0: "🟢 Normal",
        1: "🔴 Attack"
    })

    data["Attack Category"] = (
        category_predictions
    )


    # ======================================
    # COUNTS
    # ======================================

    total_records = len(data)

    normal_count = int(
        (predictions == 0).sum()
    )

    attack_count = int(
        (predictions == 1).sum()
    )

    attack_percentage = (
        attack_count / total_records
    ) * 100 if total_records > 0 else 0


    # ======================================
    # MAIN RESULTS
    # ======================================

    st.divider()

    st.header(
        "🚨 Threat Detection Results"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "📊 Total Traffic",
            total_records
        )


    with col2:

        st.metric(
            "🟢 Normal",
            normal_count
        )


    with col3:

        st.metric(
            "🔴 Attacks",
            attack_count
        )


    with col4:

        st.metric(
            "⚠️ Attack Rate",
            f"{attack_percentage:.2f}%"
        )


    # ======================================
    # RISK LEVEL
    # ======================================

    st.divider()

    st.subheader(
        "🛡️ Current Security Risk"
    )


    if attack_percentage == 0:

        risk_level = "LOW"

        st.success(
            "🟢 LOW RISK — No attacks detected."
        )


    elif attack_percentage < 20:

        risk_level = "MEDIUM"

        st.warning(
            "🟡 MEDIUM RISK — Suspicious "
            "traffic detected."
        )


    elif attack_percentage < 50:

        risk_level = "HIGH"

        st.warning(
            "🟠 HIGH RISK — Significant attack "
            "activity detected."
        )


    else:

        risk_level = "CRITICAL"

        st.error(
            "🔴 CRITICAL RISK — Large amount "
            "of attack traffic detected!"
        )


    st.metric(
        "Risk Level",
        risk_level
    )


    # ======================================
    # SECURITY ANALYTICS
    # ======================================

    st.divider()

    st.header(
        "📈 Security Analytics"
    )


    # --------------------------------------
    # NORMAL VS ATTACK
    # --------------------------------------

    st.subheader(
        "🟢 Normal vs 🔴 Attack Traffic"
    )


    traffic_data = pd.DataFrame(
        {
            "Traffic Type": [
                "Normal",
                "Attack"
            ],
            "Count": [
                normal_count,
                attack_count
            ]
        }
    )


    st.bar_chart(
        traffic_data.set_index(
            "Traffic Type"
        )
    )


    # --------------------------------------
    # ATTACK CATEGORIES
    # --------------------------------------

    st.subheader(
        "🎯 Attack Category Distribution"
    )


    attack_only = data[
        data["Prediction"] == 1
    ]


    if len(attack_only) > 0:

        category_counts = (
            attack_only[
                "Attack Category"
            ].value_counts()
        )


        st.bar_chart(
            category_counts
        )


        # Most common attack

        most_common_attack = (
            category_counts.index[0]
        )

        most_common_count = int(
            category_counts.iloc[0]
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "🎯 Most Common Attack",
                most_common_attack
            )


        with col2:

            st.metric(
                "🚨 Occurrences",
                most_common_count
            )


    else:

        st.success(
            "🟢 No attack categories detected."
        )


    # ======================================
    # SECURITY SUMMARY
    # ======================================

    st.divider()

    st.header(
        "📋 Security Summary"
    )


    summary_data = pd.DataFrame(
        {
            "Metric": [
                "Total Network Records",
                "Normal Traffic",
                "Detected Attacks",
                "Attack Rate",
                "Risk Level"
            ],

            "Value": [
                total_records,
                normal_count,
                attack_count,
                f"{attack_percentage:.2f}%",
                risk_level
            ]
        }
    )


    st.dataframe(
        summary_data,
        use_container_width=True,
        hide_index=True
    )


    # ======================================
    # DETAILED RESULTS
    # ======================================

    st.divider()

    st.header(
        "🔍 Detailed Detection Results"
    )


    st.dataframe(
        data,
        use_container_width=True
    )


    # ======================================
    # DOWNLOAD RESULTS
    # ======================================

    st.divider()

    st.header(
        "📥 Export Results"
    )


    csv_data = data.to_csv(
        index=False
    )


    st.download_button(
        label="⬇️ Download Detection Results",
        data=csv_data,
        file_name="cybersentinel_results.csv",
        mime="text/csv"
    )


# ==========================================
# NO FILE UPLOADED
# ==========================================

else:

    st.info(
        "👆 Upload a network traffic CSV "
        "file to start threat detection."
    )


    st.markdown(
        """
        ### 🛡️ How CyberSentinel Works

        1. 📁 Upload network traffic
        2. 🧹 Process the data
        3. 🔢 Encode network features
        4. 🤖 Detect Normal or Attack
        5. 🎯 Identify attack category
        6. 📊 Calculate security statistics
        7. 📈 Display analytics
        8. 📥 Export results
        """
    )
    