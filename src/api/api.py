from flask import Flask


def create_api() -> Flask:
    flask_api = Flask("api")

    from src.api.controller import prediction_api
    flask_api.register_blueprint(prediction_api)

    return flask_api
