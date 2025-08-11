"""
Simple Web Interface for Iris Flower Classifier

This script creates a web interface for the Iris classifier using Flask.
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask app
app = Flask(__name__)

# Load the Iris dataset and train the model
iris = load_iris()
X, y = iris.data, iris.target

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Iris Flower Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
        .container { text-align: center; }
        .form-group { margin: 15px 0; }
        label { display: inline-block; width: 150px; text-align: right; margin-right: 10px; }
        input { padding: 5px; width: 100px; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
        .success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .info { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Iris Flower Classifier</h1>
        <p>Enter the measurements of an iris flower to predict its species:</p>
        
        <form id="predictionForm">
            <div class="form-group">
                <label for="sepal_length">Sepal Length (cm):</label>
                <input type="number" id="sepal_length" name="sepal_length" step="0.1" required>
            </div>
            <div class="form-group">
                <label for="sepal_width">Sepal Width (cm):</label>
                <input type="number" id="sepal_width" name="sepal_width" step="0.1" required>
            </div>
            <div class="form-group">
                <label for="petal_length">Petal Length (cm):</label>
                <input type="number" id="petal_length" name="petal_length" step="0.1" required>
            </div>
            <div class="form-group">
                <label for="petal_width">Petal Width (cm):</label>
                <input type="number" id="petal_width" name="petal_width" step="0.1" required>
            </div>
            <button type="submit">Predict Species</button>
        </form>
        
        <div id="result" class="result" style="display:none;"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form data
            const formData = new FormData(this);
            const data = {
                sepal_length: parseFloat(formData.get('sepal_length')),
                sepal_width: parseFloat(formData.get('sepal_width')),
                petal_length: parseFloat(formData.get('petal_length')),
                petal_width: parseFloat(formData.get('petal_width'))
            };
            
            // Send request to the server
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.className = 'result success';
                resultDiv.innerHTML = `<h3>Prediction Result</h3>
                                      <p><strong>Predicted Species:</strong> ${data.species}</p>
                                      <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>`;
            })
            .catch(error => {
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.className = 'result info';
                resultDiv.innerHTML = `<p>Error making prediction: ${error.message}</p>`;
            });
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    """Serve the main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on input data"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract features
        features = [
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]
        
        # Make prediction
        input_data = np.array([features])
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        # Get results
        species = iris.target_names[prediction[0]]
        confidence = float(np.max(probability))
        
        return jsonify({
            'species': species,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Iris Classifier Web App...")
    print("Open your browser to http://127.0.0.1:5000")
    app.run(debug=True)
