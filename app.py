from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from passlib.context import CryptContext
import sqlite3
import os
import numpy as np
import threading
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from fastapi import UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil

app = FastAPI()

# Mount static files (ensure style.css, script.js are in the same folder)
# In production, we'll serve index.html directly from root
app.mount("/static", StaticFiles(directory="."), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_index():
    return FileResponse('index.html')

# Configuration
NB_MODEL_FILE = 'nb_model.pkl'
SVM_MODEL_FILE = 'svm_model.pkl'
VEC_FILE = 'vectorizer.pkl'
METRICS_FILE = 'model_metrics.pkl'
CSV_PATH = 'twitter_training.csv'
DB_FILE = 'users.db'

# Locks for thread-safe model updates during background retraining
model_lock = threading.Lock()
training_state = {"status": "idle", "progress": 0, "accuracy": 0, "samples": 0}

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    file_path = os.path.join(os.getcwd(), "custom_dataset.csv")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Check if the CSV is valid and has at least 2 columns
        df = pd.read_csv(file_path, nrows=2, header=None)
        if len(df.columns) < 2:
             os.remove(file_path)
             raise HTTPException(status_code=400, detail="CSV must have at least 'sentiment' and 'text' columns")
        
        # Get actual sample count
        full_df = pd.read_csv(file_path, header=None)
        return {
            "message": "Dataset uploaded successfully", 
            "filename": "custom_dataset.csv",
            "samples": len(full_df)
        }
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

@app.post("/train-custom")
async def train_custom(background_tasks: BackgroundTasks):
    file_path = os.path.join(os.getcwd(), "custom_dataset.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="No custom dataset found. Please upload first.")
        
    global training_state
    training_state = {"status": "training", "progress": 10, "accuracy": 0, "samples": 0}
    background_tasks.add_task(perform_custom_training, file_path)
    return {"message": "Custom training started"}

@app.get("/training-status")
async def get_training_status():
    return training_state

def perform_custom_training(file_path: str):
    """Robust training on custom uploaded CSV."""
    global nb_model, svm_model, vectorizer, metrics, training_state
    try:
        training_state["progress"] = 25
        # 1. Load data - handle both headered and headerless
        # Assuming the format: [text/id, entity, sentiment, text] OR [sentiment, text]
        df = pd.read_csv(file_path, header=None)
        
        # Auto-detect format
        if len(df.columns) == 4:
            df.columns = ['id', 'entity', 'sentiment', 'text']
        elif len(df.columns) == 2:
            df.columns = ['sentiment', 'text']
        else:
             # Try to find columns with string content
             raise Exception("Unsupported CSV format. Use 2 columns (sentiment, text) or 4 columns.")

        df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]
        df = df.dropna(subset=['text'])
        df['text'] = df['text'].str.lower()
        
        training_state["samples"] = len(df)
        training_state["progress"] = 40
        
        X = df['text']
        y = df['sentiment']
        
        # 2. Vectorization
        new_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
        X_vec = new_vec.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.15, random_state=42)
        
        training_state["progress"] = 60
        
        # 3. Models
        new_nb = MultinomialNB(alpha=0.1)
        new_nb.fit(X_train, y_train)
        
        base_svm = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, dual='auto')
        new_svm = CalibratedClassifierCV(base_svm, cv=3)
        new_svm.fit(X_train, y_train)
        
        training_state["progress"] = 85
        
        # 4. Metrics
        nb_acc = accuracy_score(y_test, new_nb.predict(X_test))
        svm_acc = accuracy_score(y_test, new_svm.predict(X_test))
        avg_acc = (nb_acc + svm_acc) / 2
        
        with model_lock:
            nb_model, svm_model, vectorizer = new_nb, new_svm, new_vec
            metrics = {'nb_accuracy': float(nb_acc), 'svm_accuracy': float(svm_acc)}
            joblib.dump(nb_model, NB_MODEL_FILE)
            joblib.dump(svm_model, SVM_MODEL_FILE)
            joblib.dump(vectorizer, VEC_FILE)
            joblib.dump(metrics, METRICS_FILE)
            
            # Change the default CSV for feedback to the newly uploaded one
            global CSV_PATH
            CSV_PATH = file_path
            
        training_state = {
            "status": "completed", 
            "progress": 100, 
            "accuracy": float(avg_acc), 
            "samples": len(df)
        }
    except Exception as e:
        print(f"Custom Train Error: {e}")
        training_state = {"status": "error", "progress": 0, "accuracy": 0, "error": str(e)}

def load_all_models():
    try:
        nb = joblib.load(NB_MODEL_FILE)
        svm = joblib.load(SVM_MODEL_FILE)
        vec = joblib.load(VEC_FILE)
        met = joblib.load(METRICS_FILE)
        return nb, svm, vec, met
    except Exception as e:
        print(f"Initial load failed: {e}")
        return None, None, None, None

nb_model, svm_model, vectorizer, metrics = load_all_models()

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
    conn.execute('CREATE TABLE IF NOT EXISTS feedback (text TEXT, sentiment TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
    conn.commit()
    conn.close()

init_db()

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

class User(BaseModel):
    username: str
    password: str

class SentimentRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    text: str
    sentiment: str

def perform_robust_retrain():
    """Background task to re-calibrate models with feedback oversampling."""
    global nb_model, svm_model, vectorizer, metrics
    print("Priority retraining cycle started...")
    
    try:
        # Load data
        df = pd.read_csv(CSV_PATH, header=None, names=['id', 'entity', 'sentiment', 'text'])
        df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]
        df = df.dropna(subset=['text'])
        df['text'] = df['text'].str.lower()
        
        # Split into base data and manual feedback
        # feedback rows were marked with entity='Feedback' in submit_feedback
        df_base = df[df['entity'] != 'Feedback']
        df_feedback = df[df['entity'] == 'Feedback']
        
        # OVERSAMPLING: Make user feedback 100x more influential
        if not df_feedback.empty:
            print(f"Oversampling {len(df_feedback)} manual corrections...")
            df_feedback_boosted = pd.concat([df_feedback] * 100, ignore_index=True)
            df_final = pd.concat([df_base.sample(n=min(len(df_base), 15000), random_state=42), df_feedback_boosted])
        else:
            df_final = df_base.sample(n=min(len(df_base), 20000), random_state=42)
            
        X = df_final['text']
        y = df_final['sentiment']
        
        # Vectorization (keep consistent)
        new_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
        X_vec = new_vec.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.15, random_state=42)
        
        new_nb = MultinomialNB(alpha=0.1)
        new_nb.fit(X_train, y_train)
        
        base_svm = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, dual='auto')
        new_svm = CalibratedClassifierCV(base_svm, cv=3)
        new_svm.fit(X_train, y_train)
        
        # Re-calc metrics on a clean test set
        nb_acc = accuracy_score(y_test, new_nb.predict(X_test))
        svm_acc = accuracy_score(y_test, new_svm.predict(X_test))
        
        with model_lock:
            nb_model, svm_model, vectorizer = new_nb, new_svm, new_vec
            metrics = {'nb_accuracy': float(nb_acc), 'svm_accuracy': float(svm_acc)}
            joblib.dump(nb_model, NB_MODEL_FILE)
            joblib.dump(svm_model, SVM_MODEL_FILE)
            joblib.dump(vectorizer, VEC_FILE)
            joblib.dump(metrics, METRICS_FILE)
            
        print(f"Retrain Success! New Metrics: NB={nb_acc:.2f}, SVM={svm_acc:.2f}")
    except Exception as e:
        print(f"Retrain Error: {e}")

@app.post("/signup")
def signup(user: User):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (user.username,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = pwd_context.hash(user.password)
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (user.username, hashed_password))
    conn.commit()
    conn.close()
    return {"message": "User created successfully"}

@app.post("/login")
def login(user: User):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (user.username,))
    db_user = cursor.fetchone()
    conn.close()

    if not db_user or not pwd_context.verify(user.password, db_user[1]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    return {"message": "Login successful"}

@app.post("/analyze")
def analyze_sentiment(request: SentimentRequest):
    # 1. FEEDBACK CACHE: Check if this EXACT text was manually corrected
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT sentiment FROM feedback WHERE text = ? ORDER BY timestamp DESC LIMIT 1", (request.text,))
    cache_hit = cursor.fetchone()
    conn.close()
    
    if cache_hit:
        return {
            "text": request.text,
            "results": {
                "nb": {"sentiment": cache_hit[0], "confidence": 1.0, "source": "User Feedback"},
                "svm": {"sentiment": cache_hit[0], "confidence": 1.0, "source": "User Feedback"}
            },
            "metrics": metrics
        }

    with model_lock:
        if not nb_model or not svm_model or not vectorizer:
            raise HTTPException(status_code=500, detail="Models not ready")
        
        X_vec = vectorizer.transform([request.text.lower()]) # Ensure lower
        
        nb_pred = nb_model.predict(X_vec)[0]
        nb_prob = nb_model.predict_proba(X_vec)[0]
        
        svm_pred = svm_model.predict(X_vec)[0]
        svm_prob = svm_model.predict_proba(X_vec)[0]

        return {
            "text": request.text,
            "results": {
                "nb": {"sentiment": nb_pred, "confidence": float(max(nb_prob))},
                "svm": {"sentiment": svm_pred, "confidence": float(max(svm_prob))}
            },
            "metrics": metrics
        }

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    # Normalize text
    text_clean = request.text.strip()
    
    # 1. Update CSV with 'Feedback' tag for oversampling
    # Using specific formatting to match Twitter dataset expectations
    with open(CSV_PATH, 'a', encoding='utf-8') as f:
        # ID, Entity, Sentiment, Text
        f.write(f'\n99999,Feedback,{request.sentiment},"{text_clean}"')
    
    # 2. Store in DB for instant cache hit
    conn = sqlite3.connect(DB_FILE)
    conn.execute("INSERT INTO feedback (text, sentiment) VALUES (?, ?)", (text_clean, request.sentiment))
    conn.commit()
    conn.close()
    
    # 3. Trigger retrain
    background_tasks.add_task(perform_robust_retrain)
    
    return {"message": "Success! I've learned from your feedback and will prioritize it."}

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": nb_model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
