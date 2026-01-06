# AutoJudge Flask Backend
# REST API for problem difficulty prediction

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pickle
import numpy as np
import re
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# ============================================
# Load Models at Startup
# ============================================

try:
    with open(r'D:\AutoJudge_v2\models\classification_model.pkl', 'rb') as f:
        classification_model = pickle.load(f)
    
    with open(r'D:\AutoJudge_v2\models\regression_model.pkl', 'rb') as f:
        regression_model = pickle.load(f)
    
    with open(r'D:\AutoJudge_v2\models\tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    with open(r'D:\AutoJudge_v2\models\feature_scaler.pkl', 'rb') as f:
        feature_scaler = pickle.load(f)
    
    with open(r'D:\AutoJudge_v2\models\label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    logger.info("✓ All models loaded successfully!")
    
except Exception as e:
    logger.error(f"✗ Error loading models: {str(e)}")
    logger.error("Make sure all model files are in the models directory")

# ============================================
# Feature Extraction Functions
# ============================================

def clean_text(text):
    """Clean text by removing extra whitespace"""
    if not text:
        return ''
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_manual_features(text):
    """Extract manual features from text"""
    if not text:
        return np.zeros(15)
    
    words = text.split()
    text_lower = text.lower()
    
    # Basic features
    char_count = len(text)
    word_count = len(words)
    sentence_count = len(re.findall(r'[.!?]+', text))
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    uppercase_count = sum(1 for c in text if c.isupper())
    digit_count = sum(1 for c in text if c.isdigit())
    
    # Math features
    math_symbol_count = len(re.findall(r'[+\-*/=<>≤≥≠]', text))
    equation_count = len(re.findall(r'\$.*?\$', text))
    bracket_count = len(re.findall(r'[\(\)\[\]\{\}]', text))
    dollar_sign_count = text.count('$')
    
    # Keyword features
    graph_words = ['graph', 'node', 'edge', 'tree', 'dfs', 'bfs', 'dijkstra', 'spanning']
    dp_words = ['dynamic', 'dp', 'memoization', 'optimal', 'subproblem', 'recursion']
    sort_words = ['sort', 'sorted', 'order', 'arrange', 'ascending', 'descending']
    ds_words = ['array', 'list', 'stack', 'queue', 'heap', 'hash', 'map', 'set']
    complexity_words = ['O(n)', 'O(log n)', 'complexity', 'efficient', 'optimize', 'time limit']
    
    graph_keywords = sum(text_lower.count(w) for w in graph_words)
    dp_keywords = sum(text_lower.count(w) for w in dp_words)
    sorting_keywords = sum(text_lower.count(w) for w in sort_words)
    data_structure_keywords = sum(text_lower.count(w) for w in ds_words)
    complexity_keywords = sum(text_lower.count(w) for w in complexity_words)
    
    return np.array([
        char_count, word_count, sentence_count, avg_word_length, 
        uppercase_count, digit_count, math_symbol_count, equation_count,
        bracket_count, dollar_sign_count, graph_keywords, dp_keywords,
        sorting_keywords, data_structure_keywords, complexity_keywords
    ])

def preprocess_problem(title, description, input_desc, output_desc):
    """Preprocess a problem for prediction"""
    try:
        # Clean texts
        title = clean_text(title)
        description = clean_text(description)
        input_desc = clean_text(input_desc)
        output_desc = clean_text(output_desc)
        
        # Combine text
        combined_text = f"{title} {description} {input_desc} {output_desc}"
        
        # Extract manual features
        manual_features = extract_manual_features(combined_text)
        
        # Extract TF-IDF features
        tfidf_features = tfidf_vectorizer.transform([combined_text]).toarray()[0]
        
        # Scale manual features
        manual_features_scaled = feature_scaler.transform(manual_features.reshape(1, -1))[0]
        
        # Combine features
        all_features = np.concatenate([manual_features_scaled, tfidf_features])
        
        return all_features.reshape(1, -1), combined_text
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

# ============================================
# API Endpoints
# ============================================

@app.route('/')
def home():
    """Serve the frontend HTML page"""
    try:
        # Check if frontend.html exists in the same directory
        if os.path.exists('frontend.html'):
            return send_file('frontend.html')
        else:
            # Return API info if HTML file not found
            return jsonify({
                'message': 'AutoJudge API',
                'version': '2.0',
                'status': 'running',
                'endpoints': {
                    '/': 'GET - Serve frontend HTML',
                    '/health': 'GET - Check API health',
                    '/predict': 'POST - Predict problem difficulty',
                    '/batch-predict': 'POST - Batch predict multiple problems'
                },
                'note': 'Place frontend.html in the same directory as app.py to view the UI'
            })
    except Exception as e:
        logger.error(f"Error serving frontend: {str(e)}")
        return jsonify({
            'error': 'Could not load frontend',
            'message': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'version': '2.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict difficulty for a problem"""
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate input
        required_fields = ['title', 'description', 'inputDescription', 'outputDescription']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Extract fields
        title = data['title']
        description = data['description']
        input_desc = data['inputDescription']
        output_desc = data['outputDescription']
        
        logger.info(f"Predicting difficulty for: {title}")
        
        # Preprocess
        features, combined_text = preprocess_problem(title, description, input_desc, output_desc)
        
        # Predict class
        class_encoded = classification_model.predict(features)[0]
        class_label = label_encoder.inverse_transform([class_encoded])[0]
        
        # Get class probabilities
        class_probabilities = None
        if hasattr(classification_model, 'predict_proba'):
            class_probs = classification_model.predict_proba(features)[0]
            class_probabilities = {
                label_encoder.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(class_probs)
            }
        
        # Predict score
        predicted_score = float(regression_model.predict(features)[0])
        
        # Extract some features for display
        manual_feats = extract_manual_features(combined_text)
        
        logger.info(f"Prediction complete: {class_label} (score: {predicted_score:.2f})")
        
        # Return results
        return jsonify({
            'success': True,
            'prediction': {
                'class': class_label,
                'score': round(predicted_score, 2),
                'classProbabilities': class_probabilities
            },
            'features': {
                'textLength': int(manual_feats[0]),
                'wordCount': int(manual_feats[1]),
                'mathSymbols': int(manual_feats[6]),
                'advancedKeywords': int(manual_feats[10])
            }
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Predict difficulty for multiple problems"""
    try:
        data = request.get_json()
        
        if 'problems' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing problems array'
            }), 400
        
        problems = data['problems']
        results = []
        
        logger.info(f"Batch prediction for {len(problems)} problems")
        
        for idx, problem in enumerate(problems):
            try:
                features, _ = preprocess_problem(
                    problem['title'],
                    problem['description'],
                    problem['inputDescription'],
                    problem['outputDescription']
                )
                
                class_encoded = classification_model.predict(features)[0]
                class_label = label_encoder.inverse_transform([class_encoded])[0]
                predicted_score = float(regression_model.predict(features)[0])
                
                results.append({
                    'title': problem['title'],
                    'class': class_label,
                    'score': round(predicted_score, 2)
                })
                
                logger.info(f"  [{idx+1}/{len(problems)}] {problem['title']}: {class_label}")
            
            except Exception as e:
                logger.error(f"Error processing problem {idx+1}: {str(e)}")
                results.append({
                    'title': problem.get('title', 'Unknown'),
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

# ============================================
# Error Handlers
# ============================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

# ============================================
# Run App
# ============================================

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 Starting AutoJudge API Server")
    logger.info("=" * 60)
    logger.info("📍 Server URL: http://localhost:5000")
    logger.info("📍 Local Network: http://0.0.0.0:5000")
    logger.info("")
    logger.info("Available Endpoints:")
    logger.info("  GET  /          - Frontend UI (if frontend.html exists)")
    logger.info("  GET  /health    - Health check")
    logger.info("  POST /predict   - Single prediction")
    logger.info("  POST /batch-predict - Batch prediction")
    logger.info("")
    logger.info("💡 Tip: Place frontend.html in the same directory to access the UI")
    logger.info("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)