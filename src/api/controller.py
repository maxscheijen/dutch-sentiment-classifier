import joblib

from flask import Blueprint, request, jsonify
from classifier import config

# Create Flask app
prediction_api = Blueprint("prediction_api", __name__)

# Load trained model
clf = joblib.load(config.MODEL_NAME)


# Health endpoint
@prediction_api.route("/health", methods=["GET"])
def health():
    return "Working"


# Prediction endpoint
@prediction_api.route("/predict", methods=['POST'])
def predict():
    # Get request for input data
    input_data = request.get_json(force=True)

    # Get prediction based on input data
    y_pred = clf.predict([input_data])
    output = {"sentiment": y_pred[0]}
    return jsonify(output)


# if __name__ == "__main__":
#     prediction_api.run()
