.PHONY: build up down logs ingest eval test

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

ingest:
	docker exec -it $$(docker-compose ps -q backend) python -m app.engine.ingestion ingest

eval:
	docker exec -it $$(docker-compose ps -q backend) python -m app.tests.eval_rag

test:
	docker exec -it $$(docker-compose ps -q backend) python -m pytest
