from transformers import pipeline

# Use HuggingFace pre-trained sentiment analysis
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    result = sentiment_model(text)[0]
    label = result['label'].upper()
    return label
