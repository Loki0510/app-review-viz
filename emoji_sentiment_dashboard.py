import streamlit as st
import pandas as pd
import emoji
import matplotlib.pyplot as plt
from textblob import TextBlob
import matplotlib.font_manager as fm
import os
import platform
from collections import Counter

# Load and Combine CSV Files
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

# Streamlit App Config
st.set_page_config(page_title="App Review Dashboard", layout="wide")
st.title("📊 A Prototype for Visualizing Sentiment in App Reviews Over Time")

# Sidebar Filters
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

# 📊 Overall Sentiment Distribution
st.subheader("📊 Overall Sentiment Distribution (All Versions)")
if not df.empty:
    emoji_sentiment_counts = df['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative', 'No Emoji'], fill_value=0)
    text_sentiment_counts = df['text_sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
    overall_df = pd.concat([emoji_sentiment_counts, text_sentiment_counts], axis=1)
    overall_df.columns = ["Emoji Sentiment", "Text Sentiment"]
    fig, ax = plt.subplots()
    overall_df.plot(kind="bar", ax=ax, color=["#95a5a6", "#2980b9"])
    ax.set_title("Total Reviews by Sentiment (All Versions)")
    ax.set_ylabel("Number of Reviews")
    ax.set_xlabel("Sentiment Category")
    ax.set_xticklabels(overall_df.index, rotation=45)
    ax.legend(title="Sentiment Source")
    for container in ax.containers:
        ax.bar_label(container, label_type="edge", fontsize=9)
    st.pyplot(fig)

# 📈 Sentiment Trend Over Time
st.subheader(f"📈 Sentiment Trend Over Time (Total Reviews: {len(filtered)})")
if not filtered.empty:
    trend_df = filtered.groupby([filtered['date'].dt.to_period("M"), 'sentiment']).size().unstack(fill_value=0)
    fig, ax = plt.subplots()
    trend_df.plot(ax=ax)
    ax.set_title("Sentiment Trend Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)
# 📊 Comparison of Review Content vs Emoji Sentiment
st.subheader("📊 Comparison of Review Content vs Emoji Sentiment")
if not filtered.empty:
    text_counts = filtered['text_sentiment'].value_counts().rename("Review Sentiment")
    emoji_counts = filtered['sentiment'].value_counts().rename("Emoji Sentiment")
    combined = pd.concat([text_counts, emoji_counts], axis=1).fillna(0).astype(int)
    combined = combined.reindex(['Positive', 'Neutral', 'Negative', 'No Emoji'])
    fig, ax = plt.subplots()
    combined.plot(kind='bar', stacked=False, ax=ax, color=["#2ecc71", "#9b59b6"])
    ax.set_title("Comparison of Review Content vs Emoji Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    ax.legend(title="Source")
    for container in ax.containers:
        ax.bar_label(container, label_type="edge", fontsize=10)
    st.pyplot(fig)

# 📉 Sentiment Over Time (Text vs Emoji)
st.subheader("📉 Sentiment Over Time (Text vs Emoji)")
if not filtered.empty:
    df_monthly = filtered.copy()
    df_monthly['month'] = df_monthly['date'].dt.to_period("M").dt.to_timestamp()
    monthly_summary = pd.DataFrame({
        'text_sentiment_count': df_monthly[df_monthly['text_sentiment'] != 'Neutral'].groupby('month').size(),
        'emoji_sentiment_count': df_monthly[df_monthly['sentiment'].isin(['Positive', 'Negative'])].groupby('month').size()
    }).fillna(0)
    fig, ax = plt.subplots()
    monthly_summary.plot(ax=ax, linewidth=2)
    ax.set_title("Sentiment Over Time (Text vs Emoji)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)

# 🥧 Conflicting Sentiment Pie Chart
st.subheader("🥧 Conflicting Sentiment: Text vs Emoji")
conflict_filtered = filtered[
    ((filtered['text_sentiment'] == 'Positive') & (filtered['sentiment'] == 'Negative')) |
    ((filtered['text_sentiment'] == 'Negative') & (filtered['sentiment'] == 'Positive'))
]
conflict_filtered['combo_sentiment'] = (
    "Text: " + conflict_filtered['text_sentiment'] + " | Emoji: " + conflict_filtered['sentiment']
)
conflict_counts = conflict_filtered['combo_sentiment'].value_counts()
total_reviews = len(filtered)
conflicting_reviews = len(conflict_filtered)
percentage_conflicting = (conflicting_reviews / total_reviews) * 100 if total_reviews > 0 else 0
st.write(f"🔍 **{conflicting_reviews}** out of **{total_reviews}** reviews have conflicting sentiment.")
st.write(f"📊 That's about **{percentage_conflicting:.2f}%** of the selected reviews.")
if not conflict_counts.empty:
    fig, ax = plt.subplots()
    wedges, _, autotexts = ax.pie(conflict_counts, labels=None, autopct='%1.1f%%', startangle=90)
    ax.legend(wedges, conflict_counts.index, title="Sentiment Pairs", loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("Conflicting Sentiment (Text vs Emoji)")
    ax.axis('equal')
    st.pyplot(fig)

# ✅ NEW: Aligned Sentiment Pie Chart
st.subheader("🥧 Aligned Sentiment: Text and Emoji Agreement")
aligned_filtered = filtered[
    ((filtered['text_sentiment'] == 'Positive') & (filtered['sentiment'] == 'Positive')) |
    ((filtered['text_sentiment'] == 'Negative') & (filtered['sentiment'] == 'Negative'))
]
aligned_filtered['combo_sentiment'] = (
    "Text: " + aligned_filtered['text_sentiment'] + " | Emoji: " + aligned_filtered['sentiment']
)
aligned_counts = aligned_filtered['combo_sentiment'].value_counts()
aligned_total = len(aligned_filtered)
st.write(f"✅ **{aligned_total}** reviews show alignment between text and emoji sentiment.")
if not aligned_counts.empty:
    fig, ax = plt.subplots()
    wedges, _, autotexts = ax.pie(aligned_counts, labels=None, autopct='%1.1f%%', startangle=90)
    ax.legend(wedges, aligned_counts.index, title="Sentiment Pairs", loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("Aligned Sentiment (Text and Emoji Match)")
    ax.axis('equal')
    st.pyplot(fig)

# 📝 Sample Reviews Table
st.subheader("📝 Sample Reviews")
st.dataframe(
    filtered[['date', 'review', 'sentiment', 'text_sentiment']].rename(columns={'sentiment': 'emoji_sentiment'}),
    use_container_width=True
)
# 📊 Positive Emoji Frequency Bar Plot
st.subheader("📊 Positive Emoji Frequency (Sample Reviews)")
if not filtered.empty:
    all_emojis = [e for review in filtered['review'] for e in extract_emojis(review)]
    emoji_freq = Counter(all_emojis)
    emoji_rows = []
    for emo, count in emoji_freq.items():
        sentiment = emoji_sentiment.get(emo, "").capitalize()
        if sentiment == "Positive":
            emoji_rows.append({"Emoji": emo, "Frequency": count})
    pos_emoji_df = pd.DataFrame(emoji_rows).sort_values(by="Frequency", ascending=False)
    if not pos_emoji_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(pos_emoji_df["Emoji"], pos_emoji_df["Frequency"], color="#2ecc71")
        ax.set_title("Positive Emoji Frequency")
        ax.set_xlabel("Emoji")
        ax.set_ylabel("Frequency")
        if platform.system() == "Windows":
            font_path = "C:/Windows/Fonts/seguiemj.ttf"
            if os.path.exists(font_path):
                font_prop = fm.FontProperties(fname=font_path)
                ax.set_xticklabels(pos_emoji_df['Emoji'], fontproperties=font_prop, fontsize=14)
            else:
                ax.set_xticklabels(pos_emoji_df['Emoji'], fontsize=14)
        else:
            ax.set_xticklabels(pos_emoji_df['Emoji'], fontsize=14)
        st.pyplot(fig)
    else:
        st.info("No positive emojis found in reviews.")
# 📊 Negative Emoji Frequency Bar Plot
st.subheader("📊 Negative Emoji Frequency (Sample Reviews)")
if not filtered.empty:
    emoji_rows = []
    for emo, count in emoji_freq.items():
        sentiment = emoji_sentiment.get(emo, "").capitalize()
        if sentiment == "Negative":
            emoji_rows.append({"Emoji": emo, "Frequency": count})
    neg_emoji_df = pd.DataFrame(emoji_rows).sort_values(by="Frequency", ascending=False)
    if not neg_emoji_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(neg_emoji_df["Emoji"], neg_emoji_df["Frequency"], color="#e67e22")
        ax.set_title("Negative Emoji Frequency")
        ax.set_xlabel("Emoji")
        ax.set_ylabel("Frequency")
        if platform.system() == "Windows":
            font_path = "C:/Windows/Fonts/seguiemj.ttf"
            if os.path.exists(font_path):
                font_prop = fm.FontProperties(fname=font_path)
                ax.set_xticklabels(neg_emoji_df['Emoji'], fontproperties=font_prop, fontsize=14)
            else:
                ax.set_xticklabels(neg_emoji_df['Emoji'], fontsize=14)
        else:
            ax.set_xticklabels(neg_emoji_df['Emoji'], fontsize=14)
        st.pyplot(fig)
    else:
        st.info("No negative emojis found in reviews.")
else:
    st.info("No reviews available to extract emojis.")

# 📊 Score Frequency & Average (if 'score' column exists)
st.subheader(f"📊 Frequency and Average Score for {app_selected} - version:{version_selected}")
if not filtered.empty and 'score' in filtered.columns:
    score_counts = filtered['score'].value_counts().sort_index()
    average_score = filtered['score'].mean()
    st.write(f"📉 **Average Score**: {average_score:.2f}")
    st.write("🔢 **Score Frequency Distribution**")
    st.bar_chart(score_counts)
else:
    st.info("No reviews available for the selected filters to calculate scores.")

# 🚨 NEW: Strong Conflicting Sentiment Tab (VADER powered)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
st.subheader("🆚 Strong Conflicting Sentiment (VADER)")

df_vader = filtered.copy()
df_vader['vader_sentiment'] = df_vader['review'].apply(lambda x: 'Positive' if analyzer.polarity_scores(x)['compound'] > 0.5 else ('Negative' if analyzer.polarity_scores(x)['compound'] < -0.5 else 'Neutral'))

conflicts_vader = df_vader[
    ((df_vader['vader_sentiment'] == 'Positive') & (df_vader['sentiment'] == 'Negative')) |
    ((df_vader['vader_sentiment'] == 'Negative') & (df_vader['sentiment'] == 'Positive'))
]

st.write(f"🧠 Found **{len(conflicts_vader)}** strongly conflicting reviews.")
if not conflicts_vader.empty:
    st.dataframe(conflicts_vader[['date', 'review', 'vader_sentiment', 'sentiment']], use_container_width=True)
    st.download_button("📥 Download Conflicts (CSV)", data=conflicts_vader.to_csv(index=False), file_name="vader_conflicts.csv")
else:
    st.info("No strong conflicting reviews found using VADER.")

