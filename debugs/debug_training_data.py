"""
Diagnose Training Data Issues
Run this to check what's wrong with your training data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("DIAGNOSING TRAINING DATA ISSUES")
print("="*70)

# Load your training data (the one used for training)
# Update this path to your actual file
try:
    df = pd.read_csv('D:\AutoJudge_v2\data\features_final.csv')
    print(f"✓ Loaded features_final.csv: {len(df)} samples")
except:
    try:
        df = pd.read_csv('D:\AutoJudge_v2\data\data_cleaned.csv')
        print(f"✓ Loaded data_cleaned.csv: {len(df)} samples")
    except:
        print("❌ Could not find training data file!")
        print("Please check: features_final.csv or data_cleaned.csv")
        exit()

print("\n" + "="*70)
print("ISSUE 1: CLASS DISTRIBUTION")
print("="*70)

class_dist = df['problem_class'].value_counts()
print("\nClass counts:")
for cls, count in class_dist.items():
    percentage = (count / len(df)) * 100
    print(f"  {cls:10s}: {count:5d} ({percentage:5.1f}%)")

# Check for severe imbalance
max_count = class_dist.max()
min_count = class_dist.min()
imbalance_ratio = max_count / min_count

print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print("❌ CRITICAL: Severe class imbalance!")
    print("   Your model can't learn properly with this imbalance.")
    print(f"   Largest class has {max_count} samples")
    print(f"   Smallest class has {min_count} samples")
    print("\n   SOLUTION: Undersample or use class weights")
elif imbalance_ratio > 2:
    print("⚠️  WARNING: Moderate class imbalance")
    print("   This could affect model performance")
else:
    print("✓ Class distribution is balanced")

print("\n" + "="*70)
print("ISSUE 2: SCORE DISTRIBUTION")
print("="*70)

print("\nScore statistics by class:")
score_stats = df.groupby('problem_class')['problem_score'].describe()
print(score_stats)

# Check if scores overlap too much
easy_scores = df[df['problem_class'] == 'easy']['problem_score']
medium_scores = df[df['problem_class'] == 'medium']['problem_score']
hard_scores = df[df['problem_class'] == 'hard']['problem_score']

print("\nScore ranges:")
if 'easy' in class_dist.index:
    print(f"  Easy:   {easy_scores.min():.2f} - {easy_scores.max():.2f} (mean: {easy_scores.mean():.2f})")
if 'medium' in class_dist.index:
    print(f"  Medium: {medium_scores.min():.2f} - {medium_scores.max():.2f} (mean: {medium_scores.mean():.2f})")
if 'hard' in class_dist.index:
    print(f"  Hard:   {hard_scores.min():.2f} - {hard_scores.max():.2f} (mean: {hard_scores.mean():.2f})")

# Check for overlap
if easy_scores.max() > medium_scores.min():
    overlap_easy_medium = len(easy_scores[easy_scores > medium_scores.min()])
    print(f"\n⚠️  {overlap_easy_medium} easy problems have scores > medium minimum")

if medium_scores.max() > hard_scores.min():
    overlap_medium_hard = len(medium_scores[medium_scores > hard_scores.min()])
    print(f"⚠️  {overlap_medium_hard} medium problems have scores > hard minimum")

print("\n" + "="*70)
print("ISSUE 3: FEATURE VARIANCE")
print("="*70)

# Load feature columns (exclude target columns)
feature_cols = [col for col in df.columns if col not in 
                ['problem_class', 'problem_class_encoded', 'problem_score', 
                 'title', 'description', 'input_description', 'output_description',
                 'url', 'combined_text', 'text_length', 'word_count']]

if len(feature_cols) > 0:
    features = df[feature_cols]
    
    # Check for zero variance features
    zero_var = features.var() == 0
    if zero_var.sum() > 0:
        print(f"\n❌ CRITICAL: {zero_var.sum()} features have zero variance!")
        print("   These features are useless for prediction")
        print(f"   Zero variance features: {list(features.columns[zero_var])[:10]}")
    else:
        print("✓ All features have variance")
    
    # Check for low variance features
    low_var = features.var() < 0.01
    if low_var.sum() > 10:
        print(f"\n⚠️  {low_var.sum()} features have very low variance")
        print("   These features contribute little to predictions")
else:
    print("⚠️  Could not check features (wrong file format)")

print("\n" + "="*70)
print("ISSUE 4: TRAINING SET SIZE")
print("="*70)

print(f"\nTotal samples: {len(df)}")
print(f"Samples per class:")
for cls, count in class_dist.items():
    print(f"  {cls}: {count}")

# Assuming 80-20 split
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
print(f"\nEstimated split:")
print(f"  Train: {train_size}")
print(f"  Test:  {test_size}")

for cls, count in class_dist.items():
    train_count = int(count * 0.8)
    if train_count < 50:
        print(f"\n⚠️  WARNING: Only ~{train_count} {cls} samples in training!")
        print(f"     Minimum recommended: 100 per class")

if len(df) < 300:
    print("\n❌ CRITICAL: Dataset too small!")
    print(f"   Current: {len(df)} samples")
    print(f"   Minimum recommended: 300 (100 per class)")
    print(f"   Good: 600+ (200+ per class)")

print("\n" + "="*70)
print("ISSUE 5: DATA QUALITY")
print("="*70)

# Check text lengths
if 'description' in df.columns:
    desc_lengths = df['description'].str.len()
    print(f"\nDescription length statistics:")
    print(f"  Mean: {desc_lengths.mean():.0f} chars")
    print(f"  Min:  {desc_lengths.min():.0f} chars")
    print(f"  Max:  {desc_lengths.max():.0f} chars")
    
    very_short = len(df[desc_lengths < 100])
    if very_short > len(df) * 0.1:
        print(f"\n⚠️  {very_short} problems have very short descriptions (<100 chars)")
        print("   This might indicate poor data quality")

print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Class distribution
class_dist.plot(kind='bar', ax=axes[0, 0], color=['green', 'red', 'orange'])
axes[0, 0].set_title('Class Distribution', fontweight='bold')
axes[0, 0].set_ylabel('Count')
axes[0, 0].axhline(y=len(df)/3, color='red', linestyle='--', label='Ideal (balanced)')
axes[0, 0].legend()

# 2. Score distribution by class
for cls in class_dist.index:
    data = df[df['problem_class'] == cls]['problem_score']
    axes[0, 1].hist(data, alpha=0.6, label=cls, bins=20)
axes[0, 1].set_title('Score Distribution by Class', fontweight='bold')
axes[0, 1].set_xlabel('Problem Score')
axes[0, 1].legend()

# 3. Boxplot of scores
df.boxplot(column='problem_score', by='problem_class', ax=axes[1, 0])
axes[1, 0].set_title('Score Ranges by Class', fontweight='bold')
plt.suptitle('')

# 4. Feature correlation with score
if len(feature_cols) > 0:
    # Sample some features for visualization
    sample_features = feature_cols[:10] if len(feature_cols) > 10 else feature_cols
    correlations = df[sample_features].corrwith(df['problem_score']).abs().sort_values(ascending=False)
    correlations.plot(kind='barh', ax=axes[1, 1])
    axes[1, 1].set_title('Top Feature Correlations', fontweight='bold')
    axes[1, 1].set_xlabel('Absolute Correlation with Score')

plt.tight_layout()
plt.savefig('training_data_diagnosis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization to: training_data_diagnosis.png")
plt.show()

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

issues_found = []

if imbalance_ratio > 2:
    issues_found.append(f"Class imbalance ({imbalance_ratio:.1f}:1)")

if len(df) < 500:
    issues_found.append(f"Small dataset ({len(df)} samples)")

if any(class_dist < 100):
    issues_found.append("Some classes have < 100 samples")

if issues_found:
    print("\n❌ ISSUES FOUND:")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
    
    print("\n📋 RECOMMENDED ACTIONS:")
    
    if imbalance_ratio > 2:
        print("\n1. FIX CLASS IMBALANCE:")
        print("   Option A: Undersample majority class")
        print("   Option B: Ensure class_weight='balanced' in model")
        print("   Option C: Use SMOTE for oversampling")
    
    if len(df) < 500:
        print("\n2. GET MORE DATA:")
        print(f"   Current: {len(df)} samples")
        print(f"   Target: 600+ samples (200+ per class)")
    
    if any(class_dist < 100):
        print("\n3. BALANCE CLASSES:")
        min_class = class_dist.idxmin()
        print(f"   {min_class} class needs more samples")
        print(f"   Current: {class_dist[min_class]}, Target: 100+")
else:
    print("\n✓ NO MAJOR ISSUES DETECTED")
    print("\nIf model still performs poorly, try:")
    print("  1. Increase max_depth in Random Forest")
    print("  2. Try different model (Gradient Boosting)")
    print("  3. Add more features")
    print("  4. Check for data leakage")

print("\n" + "="*70)