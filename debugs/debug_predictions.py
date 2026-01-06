"""
Debug script to test AutoJudge prediction pipeline
Run this to diagnose prediction issues
"""

import pickle
import numpy as np
import re

# ============================================
# Load All Models
# ============================================

print("="*70)
print("LOADING MODELS")
print("="*70)

with open(r'D:\AutoJudge_v2\models\classification_model.pkl', 'rb') as f:
    classification_model = pickle.load(f)
    print("✓ Classification model loaded")

with open(r'D:\AutoJudge_v2\models\regression_model.pkl', 'rb') as f:
    regression_model = pickle.load(f)
    print("✓ Regression model loaded")

with open(r'D:\AutoJudge_v2\models\tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
    print("✓ TF-IDF vectorizer loaded")

with open(r'D:\AutoJudge_v2\models\feature_scaler.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)
    print("✓ Feature scaler loaded")

with open(r'D:\AutoJudge_v2\models\label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
    print("✓ Label encoder loaded")

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
    # Clean texts
    title = clean_text(title)
    description = clean_text(description)
    input_desc = clean_text(input_desc)
    output_desc = clean_text(output_desc)
    
    # Combine text
    combined_text = f"{title} {description} {input_desc} {output_desc}"
    
    # Extract manual features
    manual_features = extract_manual_features(combined_text)
    
    print("\n" + "="*70)
    print("MANUAL FEATURES (Before Scaling)")
    print("="*70)
    feature_names = [
        'char_count', 'word_count', 'sentence_count', 'avg_word_length',
        'uppercase_count', 'digit_count', 'math_symbol_count', 'equation_count',
        'bracket_count', 'dollar_sign_count', 'graph_keywords', 'dp_keywords',
        'sorting_keywords', 'data_structure_keywords', 'complexity_keywords'
    ]
    for name, value in zip(feature_names, manual_features):
        print(f"{name:30s}: {value:10.2f}")
    
    # Extract TF-IDF features
    tfidf_features = tfidf_vectorizer.transform([combined_text]).toarray()[0]
    print(f"\n{'TF-IDF features count':30s}: {len(tfidf_features)}")
    print(f"{'TF-IDF non-zero values':30s}: {np.count_nonzero(tfidf_features)}")
    print(f"{'TF-IDF mean':30s}: {tfidf_features.mean():.6f}")
    print(f"{'TF-IDF max':30s}: {tfidf_features.max():.6f}")
    
    # Scale manual features
    manual_features_scaled = feature_scaler.transform(manual_features.reshape(1, -1))[0]
    
    print("\n" + "="*70)
    print("MANUAL FEATURES (After Scaling)")
    print("="*70)
    for name, value in zip(feature_names, manual_features_scaled):
        print(f"{name:30s}: {value:10.4f}")
    
    # Combine features
    all_features = np.concatenate([manual_features_scaled, tfidf_features])
    
    print("\n" + "="*70)
    print("COMBINED FEATURES")
    print("="*70)
    print(f"{'Total feature count':30s}: {len(all_features)}")
    print(f"{'Expected by model':30s}: {classification_model.n_features_in_}")
    print(f"{'Match':30s}: {'✓ YES' if len(all_features) == classification_model.n_features_in_ else '✗ NO - MISMATCH!'}")
    print(f"{'Features mean':30s}: {all_features.mean():.6f}")
    print(f"{'Features std':30s}: {all_features.std():.6f}")
    print(f"{'Features min':30s}: {all_features.min():.6f}")
    print(f"{'Features max':30s}: {all_features.max():.6f}")
    
    return all_features.reshape(1, -1), combined_text

# ============================================
# Test Cases
# ============================================

test_cases = {
    "EASY": {
        "title": "Watermelon",
        "description": "One hot summer day Pete and his friend Billy decided to buy a watermelon. They chose the biggest and the ripest one, in their opinion. After that the watermelon was weighed, and the scales showed w kilos. They rushed home, dying of thirst, and decided to divide the berry, however they faced a hard problem. Pete and Billy are great fans of even numbers, that's why they want to divide the watermelon in such a way that each of the two parts weighs even number of kilos, at the same time it is not obligatory that the parts are equal. The boys are extremely tired and want to start their meal as soon as possible, that's why you should help them and find out, if they can divide the watermelon in the way they want. For sure, each of them should get a part of positive weight.",
        "inputDescription": "The first (and the only) input line contains integer number w (1 ≤ w ≤ 100) — the weight of the watermelon bought by the boys.",
        "outputDescription": "Print YES, if the boys can divide the watermelon into two parts, each weighing an even number of kilos; and NO in the opposite case.",
        "expected_class": "easy"
    },
    "MEDIUM": {
        "title": "Binary Search",
        "description": "Given a sorted array of n integers, find if a target value x exists. Use binary search algorithm. The array is sorted in non-decreasing order. You must implement the O(log n) solution.",
        "inputDescription": "First line contains two integers n and x (1 ≤ n ≤ 10^5, -10^9 ≤ x ≤ 10^9). Second line contains n space-separated integers in non-decreasing order.",
        "outputDescription": "Print YES if x exists in the array, otherwise print NO.",
        "expected_class": "medium"
    },
    "HARD": {
        "title": "Maximum Flow in Graph",
        "description": "Given a directed graph with edge capacities, find the maximum flow from source to sink using Ford-Fulkerson algorithm with Edmonds-Karp implementation. The graph has n nodes and m edges. You need to implement an efficient solution with time complexity O(VE^2).",
        "inputDescription": "First line: n, m, source, sink. Next m lines: u, v, capacity for each edge (0 ≤ capacity ≤ 10^9). Nodes are 1-indexed.",
        "outputDescription": "Print a single integer - the maximum flow value from source to sink.",
        "expected_class": "hard"
    }
}

# ============================================
# Run Tests
# ============================================

print("\n\n" + "="*70)
print("TESTING PREDICTIONS")
print("="*70)

for test_name, test_data in test_cases.items():
    print("\n" + "🔍 "*35)
    print(f"TEST CASE: {test_name}")
    print(f"Expected: {test_data['expected_class'].upper()}")
    print("🔍 "*35)
    
    # Preprocess
    features, combined_text = preprocess_problem(
        test_data['title'],
        test_data['description'],
        test_data['inputDescription'],
        test_data['outputDescription']
    )
    
    # Predict class
    class_encoded = classification_model.predict(features)[0]
    class_label = label_encoder.inverse_transform([class_encoded])[0]
    
    # Get probabilities
    class_probs = classification_model.predict_proba(features)[0]
    class_probabilities = {
        label_encoder.inverse_transform([i])[0]: float(prob)
        for i, prob in enumerate(class_probs)
    }
    
    # Predict score
    predicted_score = float(regression_model.predict(features)[0])
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"{'Predicted Class':30s}: {class_label.upper()}")
    print(f"{'Predicted Score':30s}: {predicted_score:.2f}/10.0")
    print(f"{'Match Expected':30s}: {'✓ CORRECT' if class_label == test_data['expected_class'] else '✗ WRONG'}")
    
    print(f"\n{'Class Probabilities':30s}:")
    for cls, prob in sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 50)
        print(f"  {cls:10s}: {prob:6.2%} {bar}")
    
    print("\n" + "─"*70)

# ============================================
# Summary
# ============================================

print("\n\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)
print("""
If predictions are wrong, check:
1. Feature count matches (should be 515)
2. Manual features make sense (not all zeros)
3. TF-IDF features are being extracted (non-zero count > 0)
4. Class probabilities show clear winner or confused (all ~33%)
5. Predicted scores are in reasonable range (1-10)

Common Issues:
- All predictions same class → Model overfitting or data imbalance
- Very low probabilities → Feature scaling issue
- Score always similar → Regression model issue
- Feature mismatch → Re-train models with same feature engineering

Next Steps:
- If features look good but predictions wrong → Model needs retraining
- If feature count mismatch → Check feature engineering notebook
- If TF-IDF is all zeros → Vocabulary mismatch issue
""")
print("="*70)