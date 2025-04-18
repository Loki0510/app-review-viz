import streamlit as st
import pandas as pd
import emoji
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.font_manager as fm
import os
import platform
from collections import Counter

analyzer = SentimentIntensityAnalyzer()

@st.cache_data
def load_data(app_name):
    path_map = {
        "Zoom": "Cleaned_app_reviews/cleaned_zoom_reviews.csv",
        "Webex": "Cleaned_app_reviews/cleaned_webex_reviews.csv",
        "Firefox": "Cleaned_app_reviews/cleaned_firefox_reviews.csv"
    }
    df = pd.read_csv(path_map[app_name])
    df.rename(columns={'at': 'date', 'content': 'review'}, inplace=True)
    df["app"] = app_name
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['review'] = df['review'].astype(str)
    df['appVersion'] = df['appVersion'].astype(str)
    return df

# Emoji Sentiment Dictionary
emoji_sentiment = {
    '😍': 'positive', '🥰': 'positive', '😊': 'positive', '😃': 'positive', '😀': 'positive',
    '😄': 'positive', '😁': 'positive', '👍': 'positive', '💪': 'positive', '🎉': 'positive',
    '✨': 'positive', '😎': 'positive', '❤️': 'positive', '💖': 'positive', '👏': 'positive',
    '😺': 'positive', '🌟': 'positive', '🙌': 'positive', '🤩': 'positive', '👌': 'positive',
    '🥳': 'positive', '🤑': 'positive', '🤗': 'positive', '🌈': 'positive', '🍀': 'positive',
    '🏆': 'positive', '🔥': 'positive', '🫶': 'positive', '🎊': 'positive', '💫': 'positive',
    '❤': 'positive', '☺': 'positive', '♥': 'positive', '🙂': 'positive', '🙏': 'positive',
    '👍🏻': 'positive', '💯': 'positive', '⭐': 'positive', '😘': 'positive', '😂': 'positive',
    '😉': 'positive',
    '😡': 'negative', '😠': 'negative', '👎': 'negative', '😢': 'negative', '😭': 'negative',
    '😞': 'negative', '😔': 'negative', '😕': 'negative', '😩': 'negative', '😣': 'negative',
    '😫': 'negative', '😤': 'negative', '😒': 'negative', '💔': 'negative', '🙁': 'negative',
    '☹️': 'negative', '🤬': 'negative', '😰': 'negative', '😨': 'negative', '🥺': 'negative',
    '😓': 'negative', '😖': 'negative', '🤕': 'negative', '🤒': 'negative', '💢': 'negative',
    '🥶': 'negative', '😵': 'negative', '😬': 'negative', '😟': 'negative', '☹': 'negative',
    '😑': 'negative', '🙄': 'negative', '😒😒': 'negative'
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

def analyze_text_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.5:
        return 'Positive'
    elif score <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

# App Layout
st.set_page_config(page_title="App Review Dashboard", layout="wide")
st.title("📊 App Review Sentiment Dashboard")

# Tabs
dashboard_tab, conflict_tab = st.tabs(["📊 Dashboard", "🆚 Conflicting Sentiment"])

with dashboard_tab:
    st.sidebar.header("🔍 Filter Reviews")
    app_selected = st.sidebar.selectbox("Select App", ["Zoom", "Webex", "Firefox"])
    df = load_data(app_selected)
    df['emojis'] = df['review'].apply(extract_emojis)
    df['text_sentiment'] = df['review'].apply(analyze_text_sentiment)
    df['sentiment'] = df.apply(lambda row: "No Emoji" if not row['emojis'] else classify_sentiment(row['emojis']), axis=1)

    available_versions = sorted(df['appVersion'].unique(), reverse=True)
    version_selected = st.sidebar.selectbox("Select Version", available_versions)
    date_range = st.sidebar.date_input("Select Date Range", [df['date'].min(), df['date'].max()])

    filtered = df[
        (df['appVersion'] == version_selected) &
        (df['sentiment'].isin(['Positive', 'Negative', 'Neutral', 'No Emoji'])) &
        (df['date'] >= pd.to_datetime(date_range[0])) &
        (df['date'] <= pd.to_datetime(date_range[1]))
    ]

    st.subheader("📄 Sample Reviews")
    st.dataframe(filtered[['date', 'review', 'sentiment', 'text_sentiment']], use_container_width=True)

    # Add any other visualizations you had in your original dashboard here

with conflict_tab:
    st.subheader("🆚 Strong Conflicting Sentiment Reviews")
    df['text_sentiment'] = df['review'].apply(analyze_text_sentiment)
    df['emojis'] = df['review'].apply(extract_emojis)
    df['sentiment'] = df.apply(lambda row: "No Emoji" if not row['emojis'] else classify_sentiment(row['emojis']), axis=1)

    conflict_filtered = df[
        ((df['text_sentiment'] == 'Positive') & (df['sentiment'] == 'Negative')) |
        ((df['text_sentiment'] == 'Negative') & (df['sentiment'] == 'Positive'))
    ]

    def is_strong_conflict(row):
        score = analyzer.polarity_scores(row['review'])['compound']
        if score >= 0.5 and row['sentiment'] == 'Negative':
            return True
        elif score <= -0.5 and row['sentiment'] == 'Positive':
            return True
        return False

    conflict_filtered = conflict_filtered[conflict_filtered.apply(is_strong_conflict, axis=1)]
    st.write(f"Found **{len(conflict_filtered)}** strongly conflicting sentiment reviews.")
    st.dataframe(conflict_filtered[['date', 'review', 'sentiment', 'text_sentiment']], use_container_width=True)

    if len(conflict_filtered) > 0:
        st.download_button("Download CSV", data=conflict_filtered.to_csv(index=False), file_name="conflicting_reviews.csv")
