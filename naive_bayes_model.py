from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from utils import load_and_preprocess_data, create_gradio_interface

def train_naive_bayes():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, tfidf = load_and_preprocess_data()

    # Train Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("Naive Bayes Model Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(model, 'nb_model.pkl')

    return model, tfidf

if __name__ == "__main__":
    # Train model
    model, vectorizer = train_naive_bayes()
    
    # Create and launch interface
    interface = create_gradio_interface("Naive Bayes", model, vectorizer)
    interface.launch() 