# Dutch Sentiment Classifier

This Python package is able to determine the sentiment of a Dutch text. Includes a interactive web app and a REST API.

[Web App Demo](https://dutch-sentiment-classifier.herokuapp.com/)

## Install

```shell
# Install PyPy packages and local package
make install
```
After the install is completed you can run the web app or the API.

```shell
# Launch web application
make app
```

```shell
# Launch api
make api
```

## Re-train

Download and create dataset. Configuration of the model can be seen in the `src/classifier/config.py` file.

```shell
# Downloads and creates the dataset
make create_data
```

Re-train the model on the data

```shell
# Retrains the model
make train
```

Use cross-validation to asses the preformance of the model. Metrics are stored in `src/classifier/metrics/metrics.json`.

```shell
# Uses stratified cross-validation to asses preformance
make cv
```

## Dataset

The dataset contains book reviews along with associated binary sentiment polarity labels. The dataset is created by [Benjamin van der Burgh](https://github.com/benjaminvdb/110kDBRD).
