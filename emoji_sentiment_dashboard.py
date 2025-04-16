import streamlit as st
import pandas as pd
import emoji
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import matplotlib.font_manager as fm
import os
import platform
from collections import Counter

# ----------------------------------------
# Load and Combine CSV Files
# ----------------------------------------
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

# ----------------------------------------
# Emoji Sentiment Setup
# ----------------------------------------
emoji_sentiment = {
    '😍': 'positive', '🥰': 'positive', '😊': 'positive', '😃': 'positive', '😀': 'positive',
    '😄': 'positive', '😁': 'positive', '👍': 'positive', '💪': 'positive', '🎉': 'positive',
    '✨': 'positive', '😎': 'positive', '❤️': 'positive', '💖': 'positive', '👏': 'positive',
    '😺': 'positive', '🌟': 'positive', '🙌': 'positive', '🤩': 'positive', '👌': 'positive',
    '🥳': 'positive', '🤑': 'positive', '🤗': 'positive', '🌈': 'positive', '🍀': 'positive',
    '🏆': 'positive', '🔥': 'positive', '🫶': 'positive', '🎊': 'positive', '💫': 'positive',
    '❤': 'positive', '☺': 'positive', '♥': 'positive', '👍👍': 'positive', '🙂': 'positive',
    '🙏': 'positive', '👍👍👍': 'positive', '👍🏻': 'positive', '💯': 'positive', '⭐': 'positive',
    '✌': 'positive', '❣': 'positive', '👌👌': 'positive', '👍👍👍👍': 'positive', '💕': 'positive',
    '👍😊': 'positive', '😊😊': 'positive', '😉': 'positive', '👌👌👌': 'positive', '😘': 'positive',
    '😂': 'positive', '👍👍👍👍👍': 'positive', '😌': 'positive', '😍😍': 'positive',
    '😡': 'negative', '😠': 'negative', '👎': 'negative', '😢': 'negative', '😭': 'negative',
    '😞': 'negative', '😔': 'negative', '😕': 'negative', '😩': 'negative', '😣': 'negative',
    '😫': 'negative', '😤': 'negative', '😒': 'negative', '💔': 'negative', '🙁': 'negative',
    '☹️': 'negative', '🤬': 'negative', '😰': 'negative', '😨': 'negative', '🥺': 'negative',
    '😓': 'negative', '😖': 'negative', '🤕': 'negative', '🤒': 'negative', '💢': 'negative',
    '🥶': 'negative', '😵': 'negative', '😬': 'negative', '😟': 'negative', '☹': 'negative',
    '😡😡': 'negative', '😑': 'negative', '🙄': 'negative'
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
    score = TextBlob(text).sentiment.polarity
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Load data, process reviews
st.set_page_config(page_title="App Review Dashboard", layout="wide")
st.title("\U0001F4CA A Prototype for Visualizing Sentiment in App Reviews Over Time")
st.sidebar.header("\U0001F50D Filter Reviews")
app_selected = st.sidebar.selectbox("Select App", ["Zoom", "Webex", "Firefox"])
df = load_data(app_selected)
df['emojis'] = df['review'].apply(extract_emojis)
df['sentiment'] = df['emojis'].apply(classify_sentiment)
df['text_sentiment'] = df['review'].apply(analyze_text_sentiment)
available_versions = sorted(df['appVersion'].unique(), reverse=True)
version_selected = st.sidebar.selectbox("Select Version", available_versions)
date_range = st.sidebar.date_input("Select Date Range", [df['date'].min(), df['date'].max()])
filtered = df[
    (df['appVersion'] == version_selected) &
    (df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])) &
    (df['date'] >= pd.to_datetime(date_range[0])) &
    (df['date'] <= pd.to_datetime(date_range[1]))
]

# ----------------------------------------
# Bar Chart of All Emojis in emoji_sentiment Dictionary by Sentiment
# ----------------------------------------
st.subheader("\U0001F4CA Frequency of Defined Emojis Grouped by Sentiment")
emoji_category = {'Positive': [], 'Negative': [], 'Neutral': []}
for emo, sent in emoji_sentiment.items():
    sent_upper = sent.capitalize()
    if sent_upper in emoji_category:
        emoji_category[sent_upper].append(emo)
    else:
        emoji_category['Neutral'].append(emo)
emoji_counter = Counter([e for emojis in filtered['emojis'] for e in emojis])
rows = []
for category, emojis in emoji_category.items():
    for emo in emojis:
        count = emoji_counter.get(emo, 0)
        if count > 0:
            rows.append({'Emoji': emo, 'Sentiment': category, 'Count': count})
emoji_usage_df = pd.DataFrame(rows)
emoji_usage_df = emoji_usage_df.sort_values(by="Count", ascending=False).head(20)
if platform.system() == "Windows":
    emoji_font_path = "C:/Windows/Fonts/seguiemj.ttf"
    font_prop = fm.FontProperties(fname=emoji_font_path) if os.path.exists(emoji_font_path) else None
else:
    font_prop = None
fig, ax = plt.subplots(figsize=(10, 5))
for sentiment in emoji_usage_df['Sentiment'].unique():
    subset = emoji_usage_df[emoji_usage_df['Sentiment'] == sentiment]
    ax.bar(subset['Emoji'], subset['Count'], label=sentiment)
ax.set_title("Emoji Frequency by Sentiment")
ax.set_xlabel("Emoji")
ax.set_ylabel("Frequency")
if font_prop:
    ax.set_xticklabels(emoji_usage_df['Emoji'], fontproperties=font_prop, fontsize=16)
else:
    ax.set_xticklabels(emoji_usage_df['Emoji'], fontsize=16)
ax.legend(title="Sentiment")
st.pyplot(fig)

# Sample Reviews Table
st.subheader("\U0001F4DD Sample Reviews")
st.dataframe(filtered[['date', 'review', 'sentiment', 'text_sentiment']], use_container_width=True)
