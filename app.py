from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pickle
from typing import Dict

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model and vectorizer
with open("logistic_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Sentiment mapping
sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Helper function to predict sentiment
def predict_sentiment(review: str) -> str:
    try:
        # Transform the review
        review_tfidf = vectorizer.transform([review])
        
        # Predict probabilities
        prediction_proba = model.predict_proba(review_tfidf)
        predicted_class = prediction_proba.argmax(axis=1)[0]  

        # Map predicted class to sentiment
        return sentiment_mapping.get(predicted_class, "Unknown")
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error"

# API Endpoint for sentiment analysis
@app.post("/analyze")
async def analyze_sentiment(request: Request):
    try:
        data = await request.json()
        review = data.get("review")
        if not review:
            return {"error": "No review text provided"}
        sentiment = predict_sentiment(review)
        return {"sentiment": sentiment}
    except Exception as e:
        return {"error": f"Failed to process request: {str(e)}"}

@app.get("/test")
async def test_sentiment():
    review = "The product is excellent and works perfectly!"
    sentiment = predict_sentiment(review)
    return {"test_review": review, "predicted_sentiment": sentiment}
