import joblib

from flask import Flask, request, jsonify
from classifier import config

app = Flask(__name__)

clf = joblib.load(config.MODEL_NAME)


@app.route("/health", methods=["GET"])
def health():
    return "Working"


@app.route("/predict", methods=['POST'])
def predict():
    input_data = request.get_json(force=True)
    y_pred = clf.predict([input_data])
    output = {"sentiment": y_pred[0]}
    return jsonify(output)


if __name__ == "__main__":
    app.run()
