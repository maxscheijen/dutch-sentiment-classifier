.PHONY: install

install:
	pip install --upgrade pip 
	pip install -r requirements.txt

get_data:
	python3 src/dutch_sentiment_classifier/get_data.py

create_data:
	python3 src/dutch_sentiment_classifier/create_data.py