"""
Simple Iris Flower Classifier

This is a beginner-friendly machine learning project that demonstrates:
- Loading and exploring a dataset
- Training a machine learning model
- Making predictions with the trained model
"""

# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def main():
    # Load the Iris dataset
    print("Loading the Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Print dataset information
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {iris.feature_names}")
    print(f"Target classes: {iris.target_names}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train the model
    print("\nTraining the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {accuracy:.2%}")
    
    # Demonstrate making a prediction for a new flower
    print("\nMaking a prediction for a new flower...")
    # Create sample data for a new flower (sepal length, sepal width, petal length, petal width)
    new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(new_flower)
    probability = model.predict_proba(new_flower)
    
    predicted_class = iris.target_names[prediction[0]]
    confidence = np.max(probability) * 100
    
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Show what the model learned about feature importance
    print("\nFeature importance (how much each feature contributes to predictions):")
    for i, (feature, importance) in enumerate(zip(iris.feature_names, model.feature_importances_)):
        print(f"  {feature}: {importance:.2%}")

if __name__ == "__main__":
    main()
    print("\nCongratulations! You've successfully run your first machine learning project!")
