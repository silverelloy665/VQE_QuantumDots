.PHONY: all verify test benchmark hard hardware-prep round3 clean

all: verify test

verify:
	python -m pytest tests/ -q
	python scripts/20_verify_review.py
	python scripts/10_verify_audit.py

round3:
	python scripts/20_verify_review.py
	python scripts/run_hard_benchmark.py
	python scripts/run_live_hardware_job.py --dry-run
	python -m pytest tests/ -q

test:
	python -m pytest tests/ -q

hard:
	python scripts/scan_active_spaces.py
	python scripts/run_hard_benchmark.py
	python scripts/analyze_circuit_costs_hard.py

hardware-prep:
	python scripts/run_live_hardware_job.py --dry-run

benchmark:
	python run_benchmark.py

clean:
	python -c "import shutil, glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True)]"
