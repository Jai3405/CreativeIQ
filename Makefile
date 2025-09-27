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