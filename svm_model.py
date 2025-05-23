from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from utils import load_and_preprocess_data, create_gradio_interface

def train_svm():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, tfidf = load_and_preprocess_data()

    # Train SVM model
    svm_model = SVC(C=1.0, gamma='scale')
    svm_model.fit(X_train, y_train)

    # Make predictions
    y_pred = svm_model.predict(X_test)

    # Print evaluation metrics
    print("SVM Model Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(svm_model, 'svm_model.pkl')

    return svm_model, tfidf

if __name__ == "__main__":
    # Train model
    model, vectorizer = train_svm()
    
    # Create and launch interface
    interface = create_gradio_interface("SVM", model, vectorizer)
    interface.launch() 