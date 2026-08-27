.PHONY: help install test lint run clean

PYTHON ?= python3

help:
	@echo "install  install runtime and development dependencies"
	@echo "test     run the test suite"
	@echo "lint     byte-compile every module to catch syntax errors"
	@echo "run      start the scouting dashboard"
	@echo "clean    remove caches and byte-code"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install pytest

test:
	$(PYTHON) -m pytest -q tests/

lint:
	$(PYTHON) -m compileall -q *.py tests/*.py

run:
	$(PYTHON) -m streamlit run dashboard.py

clean:
	rm -rf .pytest_cache __pycache__ tests/__pycache__
	find . -name '*.pyc' -delete
