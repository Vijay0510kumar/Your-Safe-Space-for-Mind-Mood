from transformers import pipeline

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

def analyze_emotion(text):
    result = emotion_classifier(text)[0]
    emotion = result['label']
    sentiment = 'Positive' if emotion in ['joy', 'love', 'surprise'] else 'Negative'
    return emotion, sentiment
