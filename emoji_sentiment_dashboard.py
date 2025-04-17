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

# Streamlit UI
st.set_page_config(page_title="App Review Dashboard", layout="wide")
st.title("📊 A Prototype for Visualizing Sentiment in App Reviews Over Time")

st.sidebar.header("🔍 Filter Reviews")
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
# Sentiment Trend Over Time
# ----------------------------------------
st.subheader(f"📈 Sentiment Trend for {app_selected} - version: {version_selected}")


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

# Visualization 2: Text vs Emoji Sentiment Comparison
st.subheader("📊 Comparison of Review Content vs Emoji Sentiment")
if not filtered.empty:
    text_counts = filtered['text_sentiment'].value_counts().rename("Review Sentiment")
    emoji_counts = filtered['sentiment'].value_counts().rename("Emoji Sentiment")
    combined = pd.concat([text_counts, emoji_counts], axis=1).fillna(0).astype(int)
    combined = combined.reindex(['Positive', 'Neutral', 'Negative'])

    fig, ax = plt.subplots()
    combined.plot(kind='bar', stacked=False, ax=ax, color=["#2ecc71", "#9b59b6"])
    ax.set_title("Comparison of Review Content vs Emoji Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    ax.legend(title="Source")
    for container in ax.containers:
        ax.bar_label(container, label_type="edge", fontsize=10)
    st.pyplot(fig)
else:
    st.info("No data available for sentiment comparison.")

# Visualization 3: Sentiment Over Time (Text vs Emoji)
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

# Visualization 4: Conflicting Sentiment Pie Chart
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

# ✅ Visualization 5: Emoji Frequency Bar Chart
st.subheader("📊 Frequency of Defined Emojis Grouped by Sentiment")
if not filtered.empty:
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

    emoji_usage_df = pd.DataFrame(rows).sort_values(by="Count", ascending=False).head(20)

    font_prop = None
    if platform.system() == "Windows":
        font_path = "C:/Windows/Fonts/seguiemj.ttf"
        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)

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
else:
    st.info("No emoji data available for this selection.")


# ----------------------------------------
# Score Frequency & Average
# ----------------------------------------
st.subheader(f"📊 Frequency and Average Score for {app_selected} - version:{version_selected}")

if not filtered.empty:
    # Calculate frequency of scores
    score_counts = filtered['score'].value_counts().sort_index()
    
    # Calculate average score
    average_score = filtered['score'].mean()

    # Display Average Score
    st.write(f"📉 **Average Score**: {average_score:.2f}")

    # Display Frequency of Scores
    st.write("🔢 **Score Frequency Distribution**")
    
    
    st.bar_chart(score_counts)

    
else:
    st.info("No reviews available for the selected filters to calculate scores.")

# ----------------------------------------
# Sample Reviews Table
# ----------------------------------------

st.subheader("📝 Sample Reviews")
st.dataframe(filtered[['date', 'review', 'sentiment', 'text_sentiment']], use_container_width=True)
