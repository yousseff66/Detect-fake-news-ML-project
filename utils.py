import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\[[^]]*\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data():
    # Load data
    df_fake = pd.read_csv("Fake.csv")
    df_real = pd.read_csv("True.csv")

    # Add labels
    df_fake['label'] = 0
    df_real['label'] = 1
    df = pd.concat([df_fake, df_real], ignore_index=True)
    df.dropna(subset=['text'], inplace=True)

    # Clean text
    df['text'] = df['text'].apply(clean_text)

    # Split data
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    # TF-IDF transformation
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)

    # Save vectorizer
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

    return X_train, X_test, y_train, y_test, tfidf

def create_gradio_interface(model_name, model, vectorizer):
    import gradio as gr

    def classify_news(text):
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        return "FAKE" if prediction == 0 else "REAL"

    interface = gr.Interface(
        fn=classify_news,
        inputs=gr.Textbox(lines=10, placeholder="Enter the news here..."),
        outputs="text",
        title=f"Fake News Detection ({model_name})",
        description=f"Enter a news article, and we'll tell you if it's real or fake using a trained {model_name} model."
    )

    return interface 