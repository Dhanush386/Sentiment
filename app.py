from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, String, DateTime, Integer, text as sa_text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import numpy as np
import threading
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from fastapi import UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import sys
import logging

# Configure logging to stdout for Render
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files (ensure style.css, script.js are in the 'static' folder)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(BASE_DIR, 'index.html'))

# Configuration and Path Handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NB_MODEL_FILE = os.path.join(BASE_DIR, 'nb_model.pkl')
SVM_MODEL_FILE = os.path.join(BASE_DIR, 'svm_model.pkl')
VEC_FILE = os.path.join(BASE_DIR, 'vectorizer.pkl')
METRICS_FILE = os.path.join(BASE_DIR, 'model_metrics.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'twitter_training.csv')

# Database Configuration
# NOTE: The password has an @, so it's safer to use DATABASE_URL env var on Vercel
DEFAULT_DATABASE_URL = "postgresql://postgres.ucqpzgzantpgtrfammvk:Dhanush%402404@aws-1-ap-northeast-1.pooler.supabase.com:6543/postgres"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DBUser(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True, index=True)
    password = Column(String)


class DBFeedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(String)
    sentiment = Column(String)
    timestamp = Column(DateTime, server_default=sa_text("CURRENT_TIMESTAMP"))

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"Database initialization warning: {e}")

init_db()

# Locks for thread-safe model updates during background retraining
model_lock = threading.Lock()
training_state = {"status": "idle", "progress": 0, "accuracy": 0, "samples": 0}

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    # Use /tmp for temporary storage on Vercel
    file_path = os.path.join("/tmp", "custom_dataset.csv")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Check if the CSV is valid and has at least 2 columns
        df = pd.read_csv(file_path, nrows=2, header=None)
        if len(df.columns) < 2:
             os.remove(file_path)
             raise HTTPException(status_code=400, detail="CSV must have at least 'sentiment' and 'text' columns")
        
        # Get actual sample count and distribution
        full_df = pd.read_csv(file_path, header=None)
        if len(full_df.columns) == 4:
            full_df.columns = ['id', 'entity', 'sentiment', 'text']
        elif len(full_df.columns) == 2:
            full_df.columns = ['sentiment', 'text']
            
        dist = full_df['sentiment'].value_counts().to_dict()
        
        return {
            "message": "Dataset uploaded successfully", 
            "filename": "custom_dataset.csv",
            "samples": len(full_df),
            "distribution": dist
        }
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

@app.post("/train-custom")
async def train_custom(background_tasks: BackgroundTasks):
    file_path = os.path.join("/tmp", "custom_dataset.csv")
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
    """Robust training on custom uploaded CSV - FAST VERSION."""
    global nb_model, svm_model, vectorizer, metrics, training_state
    try:
        training_state["progress"] = 25
        # 1. Load data
        df = pd.read_csv(file_path, header=None)
        
        # Auto-detect format
        if len(df.columns) == 4:
            df.columns = ['id', 'entity', 'sentiment', 'text']
        elif len(df.columns) == 2:
            df.columns = ['sentiment', 'text']
        else:
             raise Exception("Unsupported CSV format. Use 2 columns (sentiment, text) or 4 columns.")

        df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]
        df = df.dropna(subset=['text'])
        df['text'] = df['text'].str.lower()
        
        # SAMPLING: Cap at 15,000 for custom data to balance speed and depth
        if len(df) > 15000:
            df = df.sample(n=15000, random_state=42)
            
        training_state["samples"] = len(df)
        training_state["progress"] = 40
        
        X = df['text']
        y = df['sentiment']
        
        # 2. Optimized Vectorization
        new_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=6000, sublinear_tf=True)
        X_vec = new_vec.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.15, random_state=42)
        
        training_state["progress"] = 60
        
        # 3. Fast SGD instead of slow calibrated SVM
        new_nb = MultinomialNB(alpha=0.1)
        new_nb.fit(X_train, y_train)
        
        new_svm = SGDClassifier(loss='modified_huber', max_iter=1000, tol=1e-3, class_weight='balanced', random_state=42)
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

try:
    nb_model, svm_model, vectorizer, metrics = load_all_models()
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"FATAL: Could not load models during startup: {e}")
    nb_model, svm_model, vectorizer, metrics = None, None, None, None

def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
        conn.execute('CREATE TABLE IF NOT EXISTS feedback (text TEXT, sentiment TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
        conn.commit()
        conn.close()
    except sqlite3.OperationalError as e:
        print(f"Database initialization warning (likely read-only on Vercel): {e}")

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
    """Background task to re-calibrate models with feedback oversampling - FAST VERSION."""
    global nb_model, svm_model, vectorizer, metrics
    print("Fast retraining cycle started...")
    
    try:
        # Load data
        df = pd.read_csv(CSV_PATH, header=None, names=['id', 'entity', 'sentiment', 'text'])
        df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]
        df = df.dropna(subset=['text'])
        df['text'] = df['text'].str.lower()
        
        # Split into base data and manual feedback
        df_base = df[df['entity'] != 'Feedback']
        df_feedback = df[df['entity'] == 'Feedback']
        
        # OVERSAMPLING + DATA CAPPING: Maintain speed even with large history
        # Cap base data at 10,000 samples for speed
        df_base_sampled = df_base.sample(n=min(len(df_base), 10000), random_state=42)
        
        if not df_feedback.empty:
            print(f"Oversampling {len(df_feedback)} manual corrections...")
            df_feedback_boosted = pd.concat([df_feedback] * 50, ignore_index=True) # 50x is enough
            df_final = pd.concat([df_base_sampled, df_feedback_boosted])
        else:
            df_final = df_base_sampled
            
        X = df_final['text']
        y = df_final['sentiment']
        
        # Optimized Vectorization (6k features is fast and accurate)
        new_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=6000, sublinear_tf=True)
        X_vec = new_vec.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.15, random_state=42)
        
        new_nb = MultinomialNB(alpha=0.1)
        new_nb.fit(X_train, y_train)
        
        # Lighter but robust SGD with Huber loss for built-in probability support
        new_svm = SGDClassifier(loss='modified_huber', max_iter=1000, tol=1e-3, class_weight='balanced', random_state=42)
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
            
        print(f"Fast Retrain Success! NB={nb_acc:.2f}, SGD={svm_acc:.2f}")
    except Exception as e:
        print(f"Retrain Error: {e}")

@app.post("/signup")
def signup(user: User):
    db = SessionLocal()
    try:
        db_user = db.query(DBUser).filter(DBUser.username == user.username).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Username already registered")
        
        hashed_password = pwd_context.hash(user.password)
        new_user = DBUser(username=user.username, password=hashed_password)
        db.add(new_user)
        db.commit()
        return {"message": "User created successfully"}
    finally:
        db.close()

@app.post("/login")
def login(user: User):
    db = SessionLocal()
    try:
        db_user = db.query(DBUser).filter(DBUser.username == user.username).first()
        if not db_user or not pwd_context.verify(user.password, db_user.password):
            raise HTTPException(status_code=400, detail="Incorrect username or password")
        
        return {"message": "Login successful"}
    finally:
        db.close()

@app.post("/analyze")
def analyze_sentiment(request: SentimentRequest):
    # 1. FEEDBACK CACHE: Check if this EXACT text was manually corrected
    db = SessionLocal()
    cache_hit = None
    try:
        cache_hit = db.query(DBFeedback).filter(DBFeedback.text == request.text).order_by(DBFeedback.timestamp.desc()).first()
    finally:
        db.close()
    
    if cache_hit:
        return {
            "text": request.text,
            "results": {
                "nb": {"sentiment": cache_hit.sentiment, "confidence": 1.0, "source": "User Feedback"},
                "svm": {"sentiment": cache_hit.sentiment, "confidence": 1.0, "source": "User Feedback"}
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
    db = SessionLocal()
    try:
        new_feedback = DBFeedback(text=text_clean, sentiment=request.sentiment)
        db.add(new_feedback)
        db.commit()
    finally:
        db.close()
    
    # 3. Trigger retrain
    background_tasks.add_task(perform_robust_retrain)
    
    return {"message": "Success! I've learned from your feedback and will prioritize it."}

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": nb_model is not None}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
