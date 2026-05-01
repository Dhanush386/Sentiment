# SentimentAI | Project Setup & Transfer Guide

This guide will help you move and set up the Sentiment Analysis project on a new laptop.

##  1. Files to Copy
Copy the entire project folder. Ensure these essential files are included:
- **Backend**: `app.py`
- **Training**: `ml_train.py`
- **Frontend**: `index.html`, `style.css`, `script.js`
- **Core Data**: `twitter_training.csv` (Required for training/feedback)
- **Trained Models**: `nb_model.pkl`, `svm_model.pkl`, `vectorizer.pkl`, `model_metrics.pkl`
- **Database**: `users.db` (Contains user accounts and feedback history)

##  2. Prerequisites
Ensure the new laptop has **Python 3.8+** installed. You can check this by running `python --version` in your terminal.

##  3. Installation Steps

1. **Open your terminal** (Command Prompt or PowerShell) inside the project folder.
2. **Install all required libraries** by running this command:
   ```bash
   pip install fastapi uvicorn scikit-learn pandas joblib passlib[pbkdf2] python-multipart
   ```

##  4. How to Run the App

1. **Start the Backend Server**:
   ```bash
   python app.py
   ```
   *The server will start at http://localhost:8000*

2. **Open the Frontend**:
   Simply open the `index.html` file in any modern web browser (Chrome, Edge, etc.).

##  5. Machine Learning Notes
- If you ever want to reset the models to their original state, delete the `.pkl` files and run:
  ```bash
  python ml_train.py
  ```
- The app supports **Online Learning**. When you provide feedback in the UI, it appends data to `twitter_training.csv` and retrains the model in the background.

## 6. How to Deploy on Render
Publish your app online in 3 easy steps:

1.  **Push to GitHub**: Create a repository on GitHub and push all your project files there.
2.  **Connect to Render**:
    - Log in to [Render](https://render.com).
    - Click **New +** > **Blueprint**.
    - Connect your GitHub repository.
3.  **Deploy (Option A: Blueprint)**: Render will automatically detect `render.yaml` and set up everything.
4.  **Deploy (Option B: Manual Web Service)**: If you've reached your Blueprint limit:
    - Click **New +** > **Web Service**.
    - Connect your GitHub repo.
    - **Runtime**: `Python 3`
    - **Build Command**: `pip install -r requirements.txt`
    - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

##  7. How to Deploy on Vercel
You can also deploy to Vercel, but with **important limitations**:

1.  **Read-Only Filesystem**: Vercel does not allow saving files. **Online Learning**, **Custom Dataset Uploads**, and **User Accounts** will not persist. The model will reset to its initial state every time the app restarts.
2.  **Deployment Steps**:
    - Push your code to GitHub (as described in section 6).
    - Log in to [Vercel](https://vercel.com).
    - Click **Add New** > **Project**.
    - Import your GitHub repository.
    - Vercel will detect `vercel.json` and deploy it automatically.

---
** Recommendation**: Use **Render** for this project. It supports a persistent filesystem which is required for the machine learning model to learn and grow over time.
