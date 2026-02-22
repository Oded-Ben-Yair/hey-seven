.PHONY: test-ci test-eval test-eval-quality lint run docker-up smoke-test ingest

test-ci:
	python3 -m pytest tests/ -v --tb=short -x --ignore=tests/test_eval.py --cov=src --cov-fail-under=90

test-eval:
	python3 -m pytest tests/test_eval.py -v --tb=short

test-eval-quality:
	python3 -m pytest tests/test_llm_judge.py tests/test_conversation_scenarios.py -v --tb=short --no-cov

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

run:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8080 --reload

docker-up:
	docker compose up --build

smoke-test:
	@curl -sf http://localhost:8080/health | python3 -m json.tool

ingest:
	python3 -c "from src.rag.pipeline import ingest_property; ingest_property()"
