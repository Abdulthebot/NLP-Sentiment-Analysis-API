from flask import Flask, request, jsonify
from transformers import pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask application
app = Flask(__name__)

# Load the sentiment analysis pipeline from Hugging Face.
# This will download and cache the model on the first run.
try:
    logging.info("Loading Hugging Face sentiment analysis model...")
    sentiment_pipeline = pipeline(
        'sentiment-analysis', 
        model='distilbert-base-uncased-finetuned-sst-2-english'
    )
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    sentiment_pipeline = None

@app.route('/')
def home():
    return "NLP Sentiment Analysis API is operational. Use the /analyze endpoint."

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    if sentiment_pipeline is None:
        return jsonify({'error': 'Model is not available or failed to load'}), 503 # Service Unavailable

    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        text_to_analyze = data.get('text', '')

        if not text_to_analyze or not isinstance(text_to_analyze, str):
            return jsonify({'error': 'Invalid input: "text" field must be a non-empty string'}), 400

        # Run the text through the pipeline
        logging.info(f"Analyzing text: '{text_to_analyze[:50]}...'")
        result = sentiment_pipeline(text_to_analyze)
        
        # The result is a list containing a dictionary, e.g., [{'label': 'POSITIVE', 'score': 0.999}]
        # We extract and clean it for a simpler response.
        if result:
            response = {
                'label': result[0]['label'],
                'score': round(result[0]['score'], 4)
            }
            logging.info(f"Analysis complete: {response}")
            return jsonify(response)
        else:
            return jsonify({'error': 'Analysis failed to produce a result'}), 500

    except Exception as e:
        logging.error(f"An error occurred during analysis: {e}")
        return jsonify({'error': f'An internal error occurred: {e}'}), 500

if __name__ == '__main__':
    # Running on port 5002 to avoid conflicts with other projects
    app.run(debug=True, port=5002)
