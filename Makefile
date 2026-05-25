.PHONY: data lint test train figures

data:
	python src/generate_mock_data.py

lint:
	ruff check .

test:
	pytest tests/

train:
	python src/train_predict.py

figures:
	python src/eda.py
	python src/business_impact.py
