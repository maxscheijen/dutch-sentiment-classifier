import joblib

from flask import Flask, request, jsonify
from classifier import config

# Create Flask app
app = Flask(__name__)

# Load trained model
clf = joblib.load(config.MODEL_NAME)


# Health endpoint
@app.route("/health", methods=["GET"])
def health():
    return "Working"


# Prediction endpoint
@app.route("/predict", methods=['POST'])
def predict():
    # Get request for input data
    input_data = request.get_json(force=True)

    # Get prediction based on input data
    y_pred = clf.predict([input_data])
    output = {"sentiment": y_pred[0]}
    return jsonify(output)


if __name__ == "__main__":
    app.run()
