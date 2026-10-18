.PHONY: all verify test benchmark

all: verify test

verify:
	python scripts/00_verify_baseline.py

test:
	python -m pytest tests/ -q

benchmark:
	python run_benchmark.py

