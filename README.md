# Iris Flower Classifier
This is a beginner-friendly machine learning project that classifies iris flowers into three species (Setosa, Versicolor, and Virginica) based on their sepal and petal measurements.

## Project Overview
The Iris dataset is a classic dataset in machine learning and statistics. It contains 150 samples from three species of iris flowers:
- Setosa
- Versicolor
- Virginica

Each sample has four features:
- Sepal length
- Sepal width
- Petal length
- Petal width

## Features

- Data exploration and visualization
- Data preprocessing and scaling
- Model training using Random Forest Classifier
- Model evaluation with accuracy metrics
- Prediction functionality for new samples

## Requirements

- Python 3.7+
- scikit-learn
- pandas
- matplotlib
- seaborn

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```
   cd iris_classifier
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the classifier:

```bash
python iris_classifier.py
```

The script will:
1. Load and explore the Iris dataset
2. Visualize the data distributions
3. Train a Random Forest model
4. Evaluate the model performance
5. Show an example prediction

### Run the Web Interface

To run the web interface:

```bash
python web_app.py
```

Then open your browser to http://127.0.0.1:5000 to use the interactive classifier.

## Project Structure

- `iris_classifier.py`: Main script containing the classifier implementation
- `simple_iris_classifier.py`: Simplified version of the classifier
- `web_app.py`: Web interface for the classifier
- `requirements.txt`: List of required Python packages
- `README.md`: Project documentation
- `CONTRIBUTING.md`: Guide for contributing to the project
- `iris_data_visualization.png`: Visualization of the dataset
- `iris_correlation_matrix.png`: Feature correlation heatmap
- `iris_confusion_matrix.png`: Model evaluation confusion matrix
- `iris_feature_importance.png`: Feature importance chart

## Learning Outcomes

By working on this project, you will learn:

- How to load and explore datasets using pandas
- Data visualization techniques with matplotlib and seaborn
- Data preprocessing and feature scaling
- How to train and evaluate machine learning models
- Model evaluation metrics and techniques
- How to make predictions with trained models

## Possible Extensions

- Try different classification algorithms (SVM, KNN, Logistic Regression)
- Implement cross-validation for more robust evaluation
- Add more visualization types
- Create a simple web interface using Flask or Streamlit
- Experiment with hyperparameter tuning

## Contributing

This project is designed for educational purposes. Feel free to:

1. Fork the repository
2. Make improvements
3. Submit pull requests

## License

This project is open source and available under the MIT License.
# Iris_Classifier
