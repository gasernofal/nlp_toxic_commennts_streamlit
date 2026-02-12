
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(
    page_title="Advanced Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==============================
# Custom CSS
# ==============================

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stMetric {
        background-color: #1C1F26;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Advanced Sentiment Analysis Dashboard")
st.markdown("Interactive dashboard for toxic comment detection")

# ==============================
# Load Data
# ==============================


@st.cache_data
def load_data():
    df = pd.read_csv("train/train.csv")
    labels = ["toxic", "severe_toxic", "obscene",
              "threat", "insult", "identity_hate"]
    df["sentiment"] = (df[labels].sum(axis=1) > 0).astype(int)
    return df


df = load_data()

model = joblib.load("sentiment_model.pkl")

# ==============================
# Sidebar
# ==============================

st.sidebar.header("🔎 Filters")

sentiment_option = st.sidebar.radio(
    "Select Sentiment",
    ["All", "Positive", "Negative"]
)

sample_size = st.sidebar.slider("Number of rows to display", 5, 100, 20)

if sentiment_option == "Positive":
    filtered_df = df[df["sentiment"] == 0]
elif sentiment_option == "Negative":
    filtered_df = df[df["sentiment"] == 1]
else:
    filtered_df = df

# ==============================
# Metrics Section
# ==============================

total = len(filtered_df)
positive = len(filtered_df[filtered_df["sentiment"] == 0])
negative = len(filtered_df[filtered_df["sentiment"] == 1])

col1, col2, col3 = st.columns(3)

col1.metric("📌 Total Comments", total)
col2.metric("😊 Positive", positive)
col3.metric("⚠ Negative", negative)

# Progress bars
st.subheader("Sentiment Ratio")

pos_ratio = positive / total if total > 0 else 0
neg_ratio = negative / total if total > 0 else 0

st.progress(pos_ratio)
st.write(f"Positive Ratio: {pos_ratio:.2%}")

st.progress(neg_ratio)
st.write(f"Negative Ratio: {neg_ratio:.2%}")

# ==============================
# Chart 1 - Pie Chart
# ==============================

st.subheader("📊 Sentiment Distribution (Pie Chart)")

fig1 = px.pie(
    names=["Positive", "Negative"],
    values=[positive, negative],
    color_discrete_sequence=["#00CC96", "#EF553B"]
)

st.plotly_chart(fig1, use_container_width=True)

# ==============================
# Chart 2 - Bar Chart
# ==============================

st.subheader("📈 Sentiment Count (Bar Chart)")

fig2 = px.bar(
    x=["Positive", "Negative"],
    y=[positive, negative],
    color=["Positive", "Negative"],
    color_discrete_sequence=["#00CC96", "#EF553B"]
)

st.plotly_chart(fig2, use_container_width=True)

# ==============================
# Chart 3 - Top Words
# ==============================

st.subheader("🔤 Most Frequent Words")

vectorizer = CountVectorizer(stop_words="english", max_features=20)
X_words = vectorizer.fit_transform(filtered_df["comment_text"])

word_counts = X_words.sum(axis=0)
words = vectorizer.get_feature_names_out()

word_df = pd.DataFrame({
    "word": words,
    "count": word_counts.A1
}).sort_values(by="count", ascending=False)

fig3 = px.bar(
    word_df,
    x="word",
    y="count"
)

st.plotly_chart(fig3, use_container_width=True)

# ==============================
# Data Table
# ==============================

st.subheader("📄 Sample Comments")
st.dataframe(filtered_df[["comment_text", "sentiment"]].head(sample_size))

# ==============================
# Prediction Section
# ==============================

st.subheader("Test New Comment")

user_input = st.text_area("Enter a comment to analyze")

if st.button("Predict"):
    prediction = model.predict([user_input])[0]
    prob = model.predict_proba([user_input])[0][prediction]

    if prediction == 1:
        st.error(f"⚠ Negative Comment (Confidence: {prob:.2%})")
    else:
        st.success(f"😊 Positive Comment (Confidence: {prob:.2%})")
