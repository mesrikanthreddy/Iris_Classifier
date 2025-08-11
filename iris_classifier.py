import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class IrisClassifier:
    def __init__(self):
        """Initialize the Iris Classifier"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_names = None
    
    def load_data(self):
        """Load the Iris dataset"""
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        # Create a DataFrame for easier handling
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.df['target'] = self.y
        self.df['species'] = [self.target_names[i] for i in self.y]
        
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Features: {self.feature_names}")
        print(f"Target classes: {self.target_names}")
        
        return self.X, self.y
    
    def explore_data(self):
        """Explore and visualize the dataset"""
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nClass distribution:")
        print(self.df['species'].value_counts())
        
        # Visualize the data
        self.visualize_data()
    
    def visualize_data(self):
        """Create visualizations for the dataset"""
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Pairplot of features
        sns.scatterplot(data=self.df, x=self.feature_names[0], y=self.feature_names[1], 
                       hue='species', ax=axes[0,0])
        axes[0,0].set_title('Sepal Length vs Sepal Width')
        
        # Plot 2: Pairplot of features
        sns.scatterplot(data=self.df, x=self.feature_names[2], y=self.feature_names[3], 
                       hue='species', ax=axes[0,1])
        axes[0,1].set_title('Petal Length vs Petal Width')
        
        # Plot 3: Distribution of sepal length
        for species in self.target_names:
            species_data = self.df[self.df['species'] == species]
            axes[1,0].hist(species_data[self.feature_names[0]], alpha=0.7, label=species)
        axes[1,0].set_xlabel(self.feature_names[0])
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Distribution of Sepal Length')
        axes[1,0].legend()
        
        # Plot 4: Distribution of petal length
        for species in self.target_names:
            species_data = self.df[self.df['species'] == species]
            axes[1,1].hist(species_data[self.feature_names[2]], alpha=0.7, label=species)
        axes[1,1].set_xlabel(self.feature_names[2])
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Distribution of Petal Length')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('iris_data_visualization.png')
        plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df.iloc[:, :-2].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.savefig('iris_correlation_matrix.png')
        plt.show()
    
    def prepare_data(self):
        """Prepare data for training"""
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
    
    def train_model(self):
        """Train the classification model"""
        # Train the model
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        self.y_train_pred = self.model.predict(self.X_train_scaled)
        self.y_test_pred = self.model.predict(self.X_test_scaled)
        
        # Calculate accuracies
        self.train_accuracy = accuracy_score(self.y_train, self.y_train_pred)
        self.test_accuracy = accuracy_score(self.y_test, self.y_test_pred)
        
        print("Model trained successfully!")
        print(f"Training Accuracy: {self.train_accuracy:.4f}")
        print(f"Test Accuracy: {self.test_accuracy:.4f}")
    
    def evaluate_model(self):
        """Evaluate the model performance"""
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_test_pred, 
                                  target_names=self.target_names))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.target_names, yticklabels=self.target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('iris_confusion_matrix.png')
        plt.show()
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.savefig('iris_feature_importance.png')
        plt.show()
        
        print("\nFeature Importance:")
        print(feature_importance)
    
    def predict_species(self, sepal_length, sepal_width, petal_length, petal_width):
        """Predict the species of an iris flower"""
        # Create input array
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Scale the input
        input_scaled = self.scaler.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)
        probability = self.model.predict_proba(input_scaled)
        
        species = self.target_names[prediction[0]]
        confidence = np.max(probability) * 100
        
        return species, confidence

def main():
    # Initialize the classifier
    classifier = IrisClassifier()
    
    # Load and explore data
    classifier.load_data()
    classifier.explore_data()
    
    # Prepare data for training
    classifier.prepare_data()
    
    # Train the model
    classifier.train_model()
    
    # Evaluate the model
    classifier.evaluate_model()
    
    # Example prediction
    print("\nExample Prediction:")
    species, confidence = classifier.predict_species(5.1, 3.5, 1.4, 0.2)
    print(f"Predicted species: {species} (Confidence: {confidence:.2f}%)")

if __name__ == "__main__":
    main()
