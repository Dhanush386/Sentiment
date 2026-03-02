import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import os

CSV_PATH = 'twitter_training.csv'

def train_models():
    print("Preparing high-accuracy initial models...")
    df = pd.read_csv(CSV_PATH, header=None, names=['id', 'entity', 'sentiment', 'text'])
    df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].str.lower()
    
    if len(df) > 30000:
        df = df.sample(n=30000, random_state=42)
    
    print(f"Dataset ready. Samples: {len(df)}")

    X = df['text']
    y = df['sentiment']

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    X_vec = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.15, random_state=42)

    # 1. NB
    print("Training Naive Bayes...")
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train, y_train)
    nb_acc = accuracy_score(y_test, nb_model.predict(X_test))

    # 2. SVM (LinearSVC with Calibration for probabilities)
    print("Training High-Accuracy Linear SVM...")
    base_svm = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, dual='auto')
    svm_model = CalibratedClassifierCV(base_svm, cv=3)
    svm_model.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm_model.predict(X_test))

    # Save
    joblib.dump(nb_model, 'nb_model.pkl')
    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    metrics = {
        'nb_accuracy': float(nb_acc),
        'svm_accuracy': float(svm_acc)
    }
    joblib.dump(metrics, 'model_metrics.pkl')

    print(f"Models created! NB Accuracy: {nb_acc:.2f}, SVM Accuracy: {svm_acc:.2f}")

if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        train_models()
    else:
        print(f"Error: {CSV_PATH} not found.")
