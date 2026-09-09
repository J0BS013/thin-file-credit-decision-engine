setup:
	python -m pip install -r requirements.txt

generate-smoke:
	python -m credit_engine --profile smoke --output-dir data/generated/smoke

test:
	python -m pytest -q
