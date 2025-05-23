import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from utils import load_and_preprocess_data

def create_cnn_model(vocab_size, max_length):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_cnn():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, tfidf = load_and_preprocess_data()
    
    # Convert sparse matrix to dense array for CNN
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
    
    # Parameters
    vocab_size = 10000
    max_length = 200
    
    # Create and train model
    model = create_cnn_model(vocab_size, max_length)
    
    # Train the model
    history = model.fit(
        X_train_dense, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.2
    )
    
    # Make predictions
    y_pred = (model.predict(X_test_dense) > 0.5).astype(int)
    
    # Print evaluation metrics
    print("CNN Model Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Save model
    model.save('cnn_model.h5')
    
    return model, tfidf

def create_gradio_interface_cnn(model, vectorizer):
    import gradio as gr
    
    def classify_news(text):
        # Transform text using TF-IDF
        vec = vectorizer.transform([text])
        # Convert to dense array
        vec_dense = vec.toarray()
        # Make prediction
        prediction = (model.predict(vec_dense) > 0.5).astype(int)[0][0]
        return "FAKE" if prediction == 0 else "REAL"
    
    interface = gr.Interface(
        fn=classify_news,
        inputs=gr.Textbox(lines=10, placeholder="Enter the news here..."),
        outputs="text",
        title="Fake News Detection (CNN)",
        description="Enter a news article, and we'll tell you if it's real or fake using a trained CNN model."
    )
    
    return interface

if __name__ == "__main__":
    # Train model
    model, vectorizer = train_cnn()
    
    # Create and launch interface
    interface = create_gradio_interface_cnn(model, vectorizer)
    interface.launch() 