import streamlit as st
import pandas as pd
import emoji
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# ✅ Page config must be the first Streamlit command
st.set_page_config(page_title="App Review Dashboard", layout="wide")

# ----------------------------------------
# Load and Combine CSV Files
# ----------------------------------------
@st.cache_data
def load_data():
    base_path = "Cleaned_app_reviews"
    zoom_df = pd.read_csv(f"{base_path}/cleaned_zoom_reviews.csv")
    webex_df = pd.read_csv(f"{base_path}/cleaned_webex_reviews.csv")
    firefox_df = pd.read_csv(f"{base_path}/cleaned_firefox_reviews.csv")

    zoom_df["app"] = "Zoom"
    webex_df["app"] = "Webex"
    firefox_df["app"] = "Firefox"

    for df in [zoom_df, webex_df, firefox_df]:
        df.rename(columns={'at': 'date', 'content': 'review'}, inplace=True)

    df = pd.concat([zoom_df, webex_df, firefox_df], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['review'] = df['review'].astype(str)
    df['appVersion'] = df['appVersion'].astype(str)
    return df

# ----------------------------------------
# Load RoBERTa Sentiment Model
# ----------------------------------------
@st.cache_resource
def load_roberta_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return tokenizer, model

tokenizer, model = load_roberta_model()
labels = ['Negative', 'Neutral', 'Positive']

# ✅ Bug fix: remove token_type_ids to avoid RoBERTa CPU crash
def analyze_text_sentiment(text):
    if not text.strip():
        return "Neutral"
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    if 'token_type_ids' in tokens:
        del tokens['token_type_ids']
    with torch.no_grad():
        output = model(**tokens)
    scores = torch.nn.functional.softmax(output.logits, dim=1).squeeze().numpy()
    return labels[np.argmax(scores)]

# ----------------------------------------
# Emoji Sentiment Setup
# ----------------------------------------
emoji_sentiment = {
    '😍': 'positive', '👍': 'positive', '💪': 'positive', '😊': 'positive', '😃': 'positive',
    '😡': 'negative', '😢': 'negative', '😭': 'negative', '👎': 'negative', '😕': 'negative'
}

def extract_emojis(text):
    return [ch for ch in text if ch in emoji.EMOJI_DATA]

def classify_sentiment(emojis):
    pos = sum(1 for e in emojis if emoji_sentiment.get(e) == 'positive')
    neg = sum(1 for e in emojis if emoji_sentiment.get(e) == 'negative')
    if pos > 0 and neg > 0:
        return 'Mixed'
    elif pos > 0:
        return 'Positive'
    elif neg > 0:
        return 'Negative'
    else:
        return 'Neutral'

# ----------------------------------------
# Streamlit UI
# ----------------------------------------
st.title("📊 App Review Emoji & Text Sentiment Comparison")

df = load_data()
df['emojis'] = df['review'].apply(extract_emojis)
df['sentiment'] = df['emojis'].apply(classify_sentiment)

with st.spinner("Analyzing text sentiment using RoBERTa..."):
    df['text_sentiment'] = df['review'].apply(analyze_text_sentiment)

# Sidebar Filters
st.sidebar.header("🔍 Filter Reviews")
app_selected = st.sidebar.selectbox("Select App", sorted(df['app'].unique()))
available_versions = sorted(df[df['app'] == app_selected]['appVersion'].unique(), reverse=True)
version_selected = st.sidebar.selectbox("Select Version", available_versions)
date_range = st.sidebar.date_input("Select Date Range", [df['date'].min(), df['date'].max()])

# Apply Filters
filtered = df[
    (df['app'] == app_selected) &
    (df['appVersion'] == version_selected) &
    (df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])) &
    (df['date'] >= pd.to_datetime(date_range[0])) &
    (df['date'] <= pd.to_datetime(date_range[1]))
]

# Sentiment Trend Over Time
st.subheader(f"📈 Sentiment Trend for {app_selected} - v{version_selected}")
if not filtered.empty:
    trend_df = filtered.groupby([filtered['date'].dt.to_period("M"), 'sentiment']).size().unstack(fill_value=0)
    fig, ax = plt.subplots()
    trend_df.plot(ax=ax)
    ax.set_title("Sentiment Trend Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)
else:
    st.warning("No reviews found for selected filters.")

# Comparison Bar Chart
st.subheader("📊 Comparison of Review Content vs Emoji Sentiment")
if not filtered.empty:
    text_counts = filtered['text_sentiment'].value_counts().rename("Review Content")
    emoji_counts = filtered['sentiment'].value_counts().rename("Emoji Sentiment")
    combined = pd.concat([text_counts, emoji_counts], axis=1).fillna(0).astype(int)

    fig, ax = plt.subplots()
    combined.plot(kind='bar', stacked=True, ax=ax, color=["#0056b3", "#66ccff"])
    ax.set_title("Comparison of Review Content vs Emoji Sentiment", fontsize=14)
    ax.set_xlabel("Sentiment Type", fontsize=12)
    ax.set_ylabel("Number of Reviews", fontsize=12)
    ax.legend(title="Sentiment Type")
    for container in ax.containers:
        ax.bar_label(container, label_type="center", fontsize=10)
    st.pyplot(fig)
else:
    st.info("No data available for sentiment comparison.")

# Sentiment Over Time (Text vs Emoji)
st.subheader("📉 Sentiment Over Time (Text vs Emoji)")
if not filtered.empty:
    df_monthly = filtered.copy()
    df_monthly['month'] = df_monthly['date'].dt.to_period("M").dt.to_timestamp()

    monthly_summary = pd.DataFrame({
        'text_sentiment_count': df_monthly[df_monthly['text_sentiment'] != 'Neutral'].groupby('month').size(),
        'emoji_sentiment_count': df_monthly[df_monthly['sentiment'] != 'Neutral'].groupby('month').size()
    }).fillna(0)

    fig, ax = plt.subplots()
    monthly_summary.plot(ax=ax, linewidth=2)
    ax.set_title("Sentiment Over Time (Text vs Emoji)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Reviews")
    ax.legend(title="Source")
    st.pyplot(fig)
else:
    st.info("📭 No data available to plot monthly sentiment trends.")

# Conflicting Sentiment Pie Chart
st.subheader("🥧 Conflicting Sentiment: Text vs Emoji")
conflict_filtered = filtered[
    ((filtered['text_sentiment'] == 'Positive') & (filtered['sentiment'] == 'Negative')) |
    ((filtered['text_sentiment'] == 'Negative') & (filtered['sentiment'] == 'Positive'))
]
conflict_filtered['combo_sentiment'] = (
    "Text: " + conflict_filtered['text_sentiment'] + " | Emoji: " + conflict_filtered['sentiment']
)
conflict_counts = conflict_filtered['combo_sentiment'].value_counts()

if not conflict_counts.empty:
    fig, ax = plt.subplots()
    wedges, _, autotexts = ax.pie(
        conflict_counts,
        labels=None,
        autopct='%1.1f%%',
        startangle=90
    )
    ax.legend(wedges, conflict_counts.index, title="Sentiment Pairs", loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("Conflicting Sentiment (Text vs Emoji)")
    ax.axis('equal')
    st.pyplot(fig)
else:
    st.info("✅ No conflicting sentiment found in current selection.")

# Most Frequent Emojis by Sentiment (with readable labels)
st.subheader("🧮 Most Frequent Emojis by Sentiment")
if not filtered.empty:
    emoji_sentiment_map = {}
    for emojis, sentiment in zip(filtered['emojis'], filtered['sentiment']):
        for e in emojis:
            if e not in emoji_sentiment_map:
                emoji_sentiment_map[e] = {'Positive': 0, 'Negative': 0, 'Mixed': 0, 'Neutral': 0}
            emoji_sentiment_map[e][sentiment] += 1

    emoji_df = pd.DataFrame.from_dict(emoji_sentiment_map, orient='index')
    top_emojis = emoji_df.sum(axis=1).sort_values(ascending=False).head(10).index
    emoji_subset = emoji_df.loc[top_emojis]

    # Convert emojis to readable labels
    emoji_labels = [emoji.demojize(e).replace(":", "").replace("_", " ").title() for e in top_emojis]

    fig, ax = plt.subplots(figsize=(10, 5))
    emoji_subset.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Top 10 Emojis Grouped by Sentiment")
    ax.set_ylabel("Count")
    ax.set_xlabel("Emoji")
    ax.set_xticks(range(len(emoji_labels)))
    ax.set_xticklabels(emoji_labels, rotation=45, ha='right', fontsize=10)
    ax.legend(title="Sentiment")
    st.pyplot(fig)
else:
    st.info("No emoji data available for this selection.")

# Sample Reviews Table
st.subheader("📝 Sample Reviews")
st.dataframe(filtered[['date', 'review', 'sentiment', 'text_sentiment']], use_container_width=True)
