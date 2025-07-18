import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)


def get_positive_advice(user_text):
    prompt = (
        f"You are a friendly mental health assistant. The user said: '{user_text}'. "
        f"Give 1 short positive advice and 1 uplifting quote. Keep it concise."
    )

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

