# CreativeIQ Makefile
# Quick commands for development and deployment

.PHONY: help setup install test run docker-build docker-run clean

# Default target
help:
	@echo "CreativeIQ - AI-Powered Design Intelligence Platform"
	@echo ""
	@echo "Available commands:"
	@echo "  setup       - Initial setup (install deps, create dirs, download models)"
	@echo "  install     - Install Python dependencies"
	@echo "  test        - Run test suite"
	@echo "  run         - Start development server"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run with Docker Compose"
	@echo "  clean       - Clean up generated files"
	@echo "  docs        - Generate documentation"
	@echo ""

# Initial setup
setup:
	@echo "Setting up CreativeIQ..."
	python scripts/setup.py

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	python scripts/run_tests.py

# Start development server
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Build Docker image
docker-build:
	docker build -t creativeiq:latest .

# Run with Docker Compose
docker-run:
	docker-compose up -d

# Stop Docker services
docker-stop:
	docker-compose down

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf uploads/*
	rm -rf logs/*

# Format code
format:
	black app/ tests/ scripts/
	isort app/ tests/ scripts/

# Lint code
lint:
	flake8 app/ tests/
	mypy app/

# Generate documentation
docs:
	@echo "API documentation available at http://localhost:8000/docs when server is running"

# Development environment
dev-setup: install
	pre-commit install
	@echo "Development environment ready!"

# Production deployment
deploy: docker-build
	@echo "Building production image..."
	docker tag creativeiq:latest creativeiq:prod
	@echo "Ready for deployment!"

# Quick test with sample image
quick-test:
	@echo "Running quick API test..."
	curl -X GET http://localhost:8000/health || echo "Server not running - start with 'make run'"

# View logs
logs:
	docker-compose logs -f creativeiq-api

# Database commands
db-init:
	@echo "Initializing databases..."
	docker-compose up -d postgres mongo redis

db-reset:
	@echo "Resetting databases..."
	docker-compose down -v
	docker-compose up -d postgres mongo redis

# Backup
backup:
	@echo "Creating backup..."
	mkdir -p backups
	tar -czf backups/creativeiq-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		app/ frontend/ data/ .env docker-compose.yml

# Monitor
monitor:
	@echo "Monitoring system resources..."
	docker stats creativeiq-creativeiq-api-1

# Security scan
security:
	safety check
	bandit -r app/

# ML Training commands
train-demo:
	python scripts/demo_training.py --mode full

train-quick:
	python scripts/demo_training.py --mode quick

train-models:
	python scripts/train_models.py --mode quick

train-production:
	python scripts/train_models.py --mode full --collect-data

retrain:
	python scripts/train_models.py --mode incremental

# Reference-Based Design Data Collection (Recommended)
collect-design-references:
	python scripts/collect_design_references.py

test-reference-fetching:
	python scripts/collect_design_references.py --test-fetch

cache-info:
	python scripts/collect_design_references.py --cache-info

preload-cache:
	python scripts/collect_design_references.py --preload 10

reference-guide:
	python scripts/collect_design_references.py --guide

# Live Visual RAG System (Real-time Design Search)
test-live-rag:
	python scripts/test_live_visual_rag.py

demo-live-rag:
	python scripts/test_live_visual_rag.py

# Legacy: Integrated Visual RAG Testing (Reference-based)
test-visual-rag:
	python scripts/test_visual_rag.py

demo-rag-system:
	python scripts/test_visual_rag.py

# Legacy Design Data Collection (Downloads)
collect-design-data:
	python scripts/collect_design_data.py

check-design-apis:
	python scripts/collect_design_data.py --check-apis

# Production deployment commands
deploy-prod:
	@echo "🚀 Deploying CreativeIQ to production..."
	chmod +x scripts/deploy.sh
	./scripts/deploy.sh deploy

deploy-status:
	./scripts/deploy.sh status

deploy-backup:
	./scripts/deploy.sh backup

deploy-logs:
	./scripts/deploy.sh logs

# Environment setup
setup-prod:
	@echo "Setting up production environment..."
	@if [ ! -f .env.prod ]; then \
		echo "⚠️  .env.prod not found. Creating from template..."; \
		cp .env.prod .env.prod.local; \
		echo "📝 Please edit .env.prod.local with your production settings"; \
	fi
	@echo "✅ Production environment ready"

# SSL setup
setup-ssl:
	@echo "🔒 Setting up SSL certificates..."
	mkdir -p nginx/ssl
	@echo "Place your SSL certificates in nginx/ssl/ as cert.pem and key.pem"
	@echo "Or run: openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem"

# Database management
db-migrate:
	docker-compose -f docker-compose.prod.yml exec creativeiq-api python -m alembic upgrade head

db-backup:
	docker-compose -f docker-compose.prod.yml exec postgres pg_dump -U creativeiq_prod creativeiq_prod > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Production health checks
health-check:
	@echo "🏥 Running production health checks..."
	@curl -f http://localhost:8000/health || echo "❌ API health check failed"
	@curl -f http://localhost:8000/api/v1/health/models || echo "❌ Models health check failed"
	@docker-compose -f docker-compose.prod.yml ps

# Performance testing
load-test:
	@echo "⚡ Running load tests..."
	@echo "Install Apache Bench: apt-get install apache2-utils"
	ab -n 100 -c 10 http://localhost:8000/health

# Monitoring
monitor-logs:
	docker-compose -f docker-compose.prod.yml logs -f

monitor-metrics:
	@echo "📊 Prometheus: http://localhost:9090"
	@echo "📈 Grafana: http://localhost:3000"