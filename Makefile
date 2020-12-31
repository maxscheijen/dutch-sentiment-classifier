.PHONY: install

install:
	pip install --upgrade pip 
	pip install -r requirements.txt

create_data:
	python3 src/classifier/dataset.py

train:
	python3 src/classifier/train.py

cv:
	python3 src/classifier/cv.py

api:
	gunicorn src.api.run:api --reload

app:
	streamlit run src/app/app.py