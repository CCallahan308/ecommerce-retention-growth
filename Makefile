.PHONY: data lint test train

data:
	python src/generate_mock_data.py

lint:
	ruff check .

test:
	pytest tests/

train:
	python src/train_predict.py
