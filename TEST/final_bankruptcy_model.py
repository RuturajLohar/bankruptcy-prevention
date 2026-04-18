import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def build_final_model():
    print("\n--- EXPANDED ROBUST MODEL EVALUATION ---")
    
    # 1. LOADING ORGANIZED DATASET
    if not os.path.exists('bankruptcy_final.csv'):
        print("Error: bankruptcy_final.csv not found. Please run data_pipeline.py first.")
        return
        
    df = pd.read_csv('bankruptcy_final.csv')
    X = df.drop('class', axis=1)
    y = df['class']
    
    # 2. MODELS TO COMPARE (Expanded list)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(algorithm='SAMME', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'Neural Network (MLP)': MLPClassifier(max_iter=500, random_state=42),
        'Perceptron': Perceptron(random_state=42)
    }
    
    # 3. CROSS-VALIDATION (10-FOLD)
    # This addresses the concern about 100% accuracy on a single split
    print("\n[Step 1] Initializing 10-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        results[name] = cv_scores
        print(f"{name}: Average CV Accuracy = {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")

    # 4. SELECTING BEST MODEL
    best_model_name = max(results, key=lambda k: results[k].mean())
    print(f"\nBest Performing Model: {best_model_name}")
    
    # 5. FINAL TRAINING & HOLD-OUT TEST
    # Split into Train and Test sets (80/20) for a final sanity check
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    final_model = models[best_model_name]
    final_model.fit(X_train, y_train)
    
    # Final Evaluation
    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n[Step 2] Final Test Accuracy ({best_model_name}): {acc*100:.2f}%")
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=['Bankruptcy', 'Non-Bankruptcy']))
    
    # 6. Save the model
    joblib.dump(final_model, 'bankruptcy_model.joblib')
    print(f"\nSUCCESS: Best model ({best_model_name}) saved as 'bankruptcy_model.joblib'")
    
    # 7. Feature Importance (Random Forest specific)
    if best_model_name == 'Random Forest':
        importances = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\n--- KEY RISK FACTORS (Feature Importance) ---")
        print(importances)

if __name__ == "__main__":
    build_final_model()
