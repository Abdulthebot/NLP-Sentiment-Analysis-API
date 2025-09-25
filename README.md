# NLP Sentiment Analysis API with Hugging Face

This project is a high-performance REST API for sentiment analysis of text. It leverages a state-of-the-art `DistilBERT` model from the Hugging Face Transformers library to classify text as **POSITIVE** or **NEGATIVE**.

![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.41-FFD21E?style=for-the-badge) ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

## The Business Problem
Companies receive vast amounts of unstructured text data from customer reviews, social media, and support tickets. This API provides a tool to automatically analyze this data, turning raw opinions into structured insights (positive/negative sentiment) that can be used to gauge customer satisfaction and inform business strategy.

## How to Run

### 1. Prerequisites
- Python 3.8+
- Pip

### 2. Installation & Execution
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/nlp-sentiment-api.git](https://github.com/YOUR_USERNAME/nlp-sentiment-api.git)
cd nlp-sentiment-api

# Install dependencies
pip install -r requirements.txt

# Run the Flask API server
# The first time you run this, it will download the ~268MB model. This is a one-time process.
python app.py
```
The server will start on `http://127.0.0.1:5002`.

## API Usage
To use the API, send a POST request to the `/analyze` endpoint with a JSON payload containing the text you want to analyze.

### Example with `curl`

**Positive Text:**
```bash
curl -X POST [http://127.0.0.1:5002/analyze](http://127.0.0.1:5002/analyze) \
-H "Content-Type: application/json" \
-d '{"text": "This product is absolutely amazing! I am incredibly happy with my purchase."}'
```
**Expected Response:**
```json
{
  "label": "POSITIVE",
  "score": 0.9999
}
```

---
**Negative Text:**
```bash
curl -X POST [http://127.0.0.1:5002/analyze](http://127.0.0.1:5002/analyze) \
-H "Content-Type: application/json" \
-d '{"text": "I am extremely disappointed with the quality. It broke after just one day."}'
```
**Expected Response:**
```json
{
  "label": "NEGATIVE",
  "score": 0.9998
}
```

## Architect's Notes
- **Transfer Learning:** Instead of training a model from scratch, we use a pre-trained `DistilBERT` model that has already learned the nuances of the English language from a massive dataset. This approach is faster, cheaper, and yields state-of-the-art results.
- **Hugging Face `pipeline`:** This project utilizes the high-level `pipeline` abstraction from the Transformers library. This is a best practice that encapsulates complex tokenization and model inference logic into a single, clean function call, making the code simple, readable, and robust.
