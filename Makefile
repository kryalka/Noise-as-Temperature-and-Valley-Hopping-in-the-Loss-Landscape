PYTHON ?= python3
REPORT_CONFIG ?= configs/pipeline/report_example.yaml
DIAGNOSTIC_CONFIG ?= configs/pipeline/diagnostic_pairs_example.yaml

.PHONY: install test smoke report-flow diagnostic

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -m pytest -q tests/test_cli.py tests/test_pipeline_smoke.py

report-flow:
	$(PYTHON) -m ntempvh.pipeline.report_flow --config $(REPORT_CONFIG) --python_bin $(PYTHON)

diagnostic:
	$(PYTHON) -m ntempvh.pipeline.diagnostic_pipeline --config $(DIAGNOSTIC_CONFIG)
