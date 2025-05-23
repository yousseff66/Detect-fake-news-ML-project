from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from utils import load_and_preprocess_data, create_gradio_interface

def train_knn():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, tfidf = load_and_preprocess_data()

    # Train KNN model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("KNN Model Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(model, 'knn_model.pkl')

    return model, tfidf

if __name__ == "__main__":
    # Train model
    model, vectorizer = train_knn()
    
    # Create and launch interface
    interface = create_gradio_interface("KNN", model, vectorizer)
    interface.launch() 