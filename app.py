import streamlit as st
import pandas as pd
import datetime
import os
from transformers import pipeline
from groq import Groq
from dotenv import load_dotenv
import plotly.express as px

# ---------------- ENVIRONMENT SETUP ----------------
load_dotenv()

# Initialize Groq API Key Correctly
groq_api_key = os.getenv("GROQ_API_KEY")

# Create Groq Client
groq_client = Groq(api_key=groq_api_key)




# Hugging Face Sentiment Pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# ---------------- PATHS ----------------
DATA_PATH = 'data/user_logs.csv'
BEST_MOMENTS_PATH = 'data/best_moments.csv'


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Your Safe Space - Mood Journal", layout="wide")


# ---------------- CUSTOM STYLING ----------------
st.markdown(
    """
    <style>
        .stApp {
            background: url('https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1470&q=80');
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: #2D3142;
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            padding-top: 2rem;
            background-color: rgba(255, 255, 255, 0.75);
            border-radius: 16px;
            padding: 2rem;
        }
        h1 {
            color: #2D3142;
        }
        h2, h3, h4 {
            color: #4F6D7A;
        }
        .stTextInput>div>div>input {
            background-color: #f0f4f8;
            color: #2D3142;
        }
        .stTextArea textarea {
            background-color: #f0f4f8;
            color: #2D3142;
        }
        .stButton>button {
            background-color: #88BDBC;
            color: #2D3142;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .stDataFrame {
            background-color: #f0f4f8;
            color: #2D3142;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align: center; color: #2D3142;'>🌊 Your Safe Space for Mind & Mood</h1>", unsafe_allow_html=True)


# ---------------- FUNCTION FOR POSITIVE ADVICE ----------------
def get_positive_advice(user_text):
    prompt = (
        f"You are a friendly mental health assistant. The user said: '{user_text}'. "
        f"Give 1 short positive advice and 1 uplifting quote. Keep it concise."
    )
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


# ---------------- JOURNAL ENTRY ----------------
st.markdown("## 📝 Add Your Thoughts Below")
today_date = st.date_input("Date", datetime.date.today())
journal_entry = st.text_area("How do you feel today? Write freely, this space is just for you:")

if st.button("Analyze & Save"):
    sentiment_result = sentiment_pipeline(journal_entry)[0]
    sentiment = sentiment_result['label']
    emotion = "Positive" if sentiment == "POSITIVE" else "Negative"

    advice_response = get_positive_advice(journal_entry)

    new_data = pd.DataFrame({
        'date': [today_date],
        'entry': [journal_entry],
        'emotion': [emotion],
        'sentiment': [sentiment],
        'week': [today_date.isocalendar()[1]],
        'month': [today_date.month]
    })

    if os.path.exists(DATA_PATH):
        existing_data = pd.read_csv(DATA_PATH)
        all_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        all_data = new_data

    all_data.to_csv(DATA_PATH, index=False)
    st.success("✅ Your entry has been saved.")

    st.markdown(f"###  Detected Emotion: **{emotion}** ({sentiment})")
    st.markdown(f"#### 🌟 LLM Advice & Quote: {advice_response}")


# ---------------- BEST MOMENTS (Collapsible) ----------------
st.markdown("---")
with st.expander(" Your Best Moments Collection (Click to View / Add)"):
    best_moment = st.text_area("Write down any happy memory you'd like to keep forever:")

    if st.button("Save Best Moment"):
        new_moment = pd.DataFrame({
            'date': [datetime.date.today()],
            'moment_description': [best_moment]
        })

        if os.path.exists(BEST_MOMENTS_PATH):
            moments_data = pd.read_csv(BEST_MOMENTS_PATH)
            moments_data = pd.concat([moments_data, new_moment], ignore_index=True)
        else:
            moments_data = new_moment

        moments_data.to_csv(BEST_MOMENTS_PATH, index=False)
        st.success("🎉 Your memory has been saved to Best Moments.")

    if os.path.exists(BEST_MOMENTS_PATH):
        best_data = pd.read_csv(BEST_MOMENTS_PATH)
        st.dataframe(best_data.sort_values(by='date', ascending=False), use_container_width=True)
    else:
        st.info("No moments saved yet. Start collecting happy memories!")


# ---------------- REFLECTIONS (Collapsible) ----------------
st.markdown("---")
with st.expander(" View All Your Reflections (Click to Expand)"):
    if os.path.exists(DATA_PATH):
        data = pd.read_csv(DATA_PATH)
        data['date'] = pd.to_datetime(data['date'])
        st.markdown("###  All Your Reflections")
        st.dataframe(data[['date', 'entry', 'emotion', 'sentiment']].sort_values(by='date', ascending=False), use_container_width=True)
    else:
        st.info("No reflections saved yet. Start today!")


# ---------------- MODERN DASHBOARD VISUALS ----------------
if os.path.exists(DATA_PATH):
    st.markdown("---")
    st.markdown("## 📊 Mood Over Time (Timeline)")
    fig = px.area(
        data,
        x='date',
        y='emotion',
        markers=True,
        title="Your Emotional Journey",
    )
    fig.update_layout(
        plot_bgcolor='rgba(240, 244, 248, 0.7)',
        paper_bgcolor='rgba(240, 244, 248, 0.7)',
        font_color='#2D3142',
        title_font_color='#4F6D7A',
        xaxis=dict(color='#4F6D7A'),
        yaxis=dict(color='#4F6D7A')
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------- Emotion Frequency Bar Chart ----------------
    st.markdown("### 📈 Emotion Frequency Tracker")
    emotion_counts = data['emotion'].value_counts().reset_index()
    emotion_counts.columns = ['Emotion', 'Count']
    fig_bar = px.bar(emotion_counts, x='Emotion', y='Count',
                     color='Emotion',
                     color_discrete_sequence=px.colors.sequential.Teal,
                     title='Emotion Frequency')
    fig_bar.update_layout(plot_bgcolor='rgba(240, 244, 248, 0.7)', paper_bgcolor='rgba(240, 244, 248, 0.7)')
    st.plotly_chart(fig_bar, use_container_width=True)

    # ------------- Sentiment Pie Chart ----------------
    st.markdown("###  Sentiment Distribution")
    sentiment_counts = data['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig_pie = px.pie(sentiment_counts, names='Sentiment', values='Count',
                     color_discrete_sequence=px.colors.sequential.Teal,
                     title='Sentiment Distribution')
    fig_pie.update_layout(paper_bgcolor='rgba(240, 244, 248, 0.7)')
    st.plotly_chart(fig_pie, use_container_width=True)

    # ------------- Weekly Analysis ----------------
    st.markdown("### 📆 Weekly Emotion Overview")
    weekly_data = data.groupby('week')['emotion'].apply(lambda x: x.value_counts().idxmax()).reset_index()
    weekly_data.columns = ['Week', 'Most Common Emotion']
    st.dataframe(weekly_data)

    # ------------- Monthly Analysis ----------------
    st.markdown("### 📆 Monthly Emotion Overview")
    monthly_data = data.groupby('month')['emotion'].apply(lambda x: x.value_counts().idxmax()).reset_index()
    monthly_data.columns = ['Month', 'Most Common Emotion']
    st.dataframe(monthly_data)
