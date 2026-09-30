# FP&A rolling forecast - common tasks
AS_OF ?= $(shell uv run python -c "import yaml;print(yaml.safe_load(open('config/settings.yaml'))['as_of'])")

.PHONY: setup data forecast pipeline app test lint notebooks clean

setup:  ## install Python 3.12 + dependencies
	uv sync

data:  ## generate ERP extracts, validate, build DuckDB marts
	uv run python -m fpa.pipeline run --as-of $(AS_OF) --stages generate,transform

forecast:  ## model tournament, variance flags and Excel pack (needs `make data`)
	uv run python -m fpa.pipeline run --as-of $(AS_OF) --stages forecast,variance,report

pipeline:  ## full monthly close: generate -> transform -> forecast -> variance -> report
	uv run python -m fpa.pipeline run --as-of $(AS_OF)

app:  ## launch the Streamlit dashboard locally
	uv run streamlit run app/streamlit_app.py

test:
	uv run pytest

lint:
	uv run ruff check src tests app

notebooks:  ## re-execute the analysis notebooks in place
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

clean:
	rm -rf data/raw data/warehouse .pytest_cache
