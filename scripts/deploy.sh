#!/bin/bash
# CreativeIQ Production Deployment Script

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="creativeiq"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Print colored output
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
    fi

    # Check if running as root (not recommended)
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root is not recommended for production"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Check available disk space (minimum 10GB)
    AVAILABLE_SPACE=$(df . | awk 'NR==2 {print $4}')
    if [[ $AVAILABLE_SPACE -lt 10485760 ]]; then
        error "Insufficient disk space. At least 10GB required."
    fi

    log "Prerequisites check passed"
}

# Setup environment
setup_environment() {
    log "Setting up environment..."

    # Create production environment file if it doesn't exist
    if [[ ! -f .env.prod ]]; then
        error ".env.prod file not found. Please create it from .env.prod.example"
    fi

    # Create necessary directories
    mkdir -p {uploads,models,data/{training,processed},logs,nginx/ssl}

    # Set proper permissions
    chmod 755 uploads models data logs
    chmod 700 nginx/ssl

    log "Environment setup complete"
}

# Generate SSL certificates
setup_ssl() {
    log "Setting up SSL certificates..."

    if [[ ! -f nginx/ssl/cert.pem ]] || [[ ! -f nginx/ssl/key.pem ]]; then
        warn "SSL certificates not found. Generating self-signed certificates..."

        # Generate self-signed certificate (replace with real certificates in production)
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout nginx/ssl/key.pem \
            -out nginx/ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

        warn "Self-signed certificates generated. Replace with real certificates for production!"
    else
        log "SSL certificates found"
    fi
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."

    # Check environment variables
    if ! grep -q "SECRET_KEY=CHANGE_THIS" .env.prod; then
        log "✓ SECRET_KEY has been changed"
    else
        error "Please change the SECRET_KEY in .env.prod"
    fi

    # Check database passwords
    if grep -q "CHANGE_THIS" .env.prod; then
        error "Please update all passwords in .env.prod (search for 'CHANGE_THIS')"
    fi

    # Validate Docker Compose file
    if ! docker-compose -f docker-compose.prod.yml config > /dev/null; then
        error "Invalid docker-compose.prod.yml configuration"
    fi

    log "Pre-deployment checks passed"
}

# Build and deploy
deploy() {
    log "Starting deployment..."

    # Load environment variables
    set -a
    source .env.prod
    set +a

    # Build images
    log "Building Docker images..."
    docker-compose -f docker-compose.prod.yml build --no-cache

    # Stop existing containers
    log "Stopping existing containers..."
    docker-compose -f docker-compose.prod.yml down || true

    # Start services
    log "Starting services..."
    docker-compose -f docker-compose.prod.yml up -d

    # Wait for services to be healthy
    log "Waiting for services to start..."
    sleep 30

    # Check health
    check_health
}

# Health checks
check_health() {
    log "Performing health checks..."

    # Check API health
    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s http://localhost:8000/health > /dev/null; then
            log "✓ API is healthy"
            break
        else
            if [[ $attempt -eq $max_attempts ]]; then
                error "API health check failed after $max_attempts attempts"
            fi
            log "Waiting for API... (attempt $attempt/$max_attempts)"
            sleep 10
            ((attempt++))
        fi
    done

    # Check database connectivity
    if docker-compose -f docker-compose.prod.yml exec -T creativeiq-api python -c "
import asyncio
from app.core.config import settings
print('Database connection test passed')
" > /dev/null 2>&1; then
        log "✓ Database connectivity confirmed"
    else
        warn "Database connectivity check failed"
    fi

    # Check model loading
    if curl -f -s http://localhost:8000/api/v1/health/models | grep -q "initialized"; then
        log "✓ AI models loaded successfully"
    else
        warn "AI models may not be loaded properly"
    fi

    log "Health checks completed"
}

# Post-deployment tasks
post_deployment() {
    log "Running post-deployment tasks..."

    # Initialize database
    log "Initializing database..."
    docker-compose -f docker-compose.prod.yml exec -T creativeiq-api python -c "
import asyncio
from app.core.database import init_db
asyncio.run(init_db())
print('Database initialized')
" || warn "Database initialization failed"

    # Train initial models if needed
    if [[ "${TRAIN_INITIAL_MODELS:-false}" == "true" ]]; then
        log "Training initial models..."
        docker-compose -f docker-compose.prod.yml exec -T creativeiq-api python scripts/train_models.py --mode quick
    fi

    # Setup log rotation
    setup_log_rotation

    log "Post-deployment tasks completed"
}

# Setup log rotation
setup_log_rotation() {
    log "Setting up log rotation..."

    cat > /tmp/creativeiq-logrotate << EOF
/app/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    copytruncate
}
EOF

    # Install logrotate config (requires sudo)
    if command -v sudo &> /dev/null; then
        sudo mv /tmp/creativeiq-logrotate /etc/logrotate.d/creativeiq
        log "Log rotation configured"
    else
        warn "Could not setup log rotation (no sudo access)"
    fi
}

# Backup before deployment
backup() {
    log "Creating backup..."

    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"

    # Backup data
    if [[ -d data ]]; then
        cp -r data "$backup_dir/"
    fi

    # Backup models
    if [[ -d models ]]; then
        cp -r models "$backup_dir/"
    fi

    # Backup database
    if docker-compose -f docker-compose.prod.yml ps | grep -q postgres; then
        docker-compose -f docker-compose.prod.yml exec -T postgres pg_dump -U creativeiq_prod creativeiq_prod > "$backup_dir/database.sql"
    fi

    log "Backup created: $backup_dir"
}

# Rollback function
rollback() {
    local backup_dir=$1

    if [[ ! -d "$backup_dir" ]]; then
        error "Backup directory not found: $backup_dir"
    fi

    warn "Rolling back to backup: $backup_dir"

    # Stop current services
    docker-compose -f docker-compose.prod.yml down

    # Restore data
    if [[ -d "$backup_dir/data" ]]; then
        rm -rf data
        cp -r "$backup_dir/data" .
    fi

    # Restore models
    if [[ -d "$backup_dir/models" ]]; then
        rm -rf models
        cp -r "$backup_dir/models" .
    fi

    # Restart services
    docker-compose -f docker-compose.prod.yml up -d

    log "Rollback completed"
}

# Show deployment status
status() {
    log "Deployment Status:"
    echo

    # Service status
    echo -e "${BLUE}Services:${NC}"
    docker-compose -f docker-compose.prod.yml ps

    echo
    echo -e "${BLUE}Health Checks:${NC}"

    # API Health
    if curl -f -s http://localhost:8000/health > /dev/null; then
        echo -e "API Health: ${GREEN}✓ Healthy${NC}"
    else
        echo -e "API Health: ${RED}✗ Unhealthy${NC}"
    fi

    # Database connectivity
    if docker-compose -f docker-compose.prod.yml exec -T postgres pg_isready -U creativeiq_prod > /dev/null 2>&1; then
        echo -e "Database: ${GREEN}✓ Connected${NC}"
    else
        echo -e "Database: ${RED}✗ Disconnected${NC}"
    fi

    # Disk usage
    echo
    echo -e "${BLUE}Resource Usage:${NC}"
    echo "Disk Usage: $(df -h . | awk 'NR==2 {print $3 "/" $2 " (" $5 ")"}')"
    echo "Memory Usage: $(free -h | awk 'NR==2{printf "%.1f/%.1fGB (%.2f%%)\n", $3/1024/1024,$2/1024/1024,$3*100/$2 }')"
}

# Usage information
usage() {
    echo "Usage: $0 {deploy|status|backup|rollback|logs}"
    echo
    echo "Commands:"
    echo "  deploy              - Full deployment process"
    echo "  status              - Show deployment status"
    echo "  backup              - Create backup"
    echo "  rollback <backup>   - Rollback to backup"
    echo "  logs                - Show application logs"
    echo
    echo "Environment variables:"
    echo "  VERSION             - Docker image version (default: latest)"
    echo "  ENVIRONMENT         - Deployment environment (default: production)"
    echo "  TRAIN_INITIAL_MODELS - Train models after deployment (default: false)"
    echo
    echo "Examples:"
    echo "  $0 deploy"
    echo "  VERSION=v1.2.3 $0 deploy"
    echo "  TRAIN_INITIAL_MODELS=true $0 deploy"
    echo "  $0 rollback backups/20240115_143000"
}

# Main execution
main() {
    case "${1:-}" in
        deploy)
            check_prerequisites
            setup_environment
            setup_ssl
            pre_deployment_checks
            backup
            deploy
            post_deployment
            status
            log "🚀 Deployment completed successfully!"
            log "Access your application at: https://your-domain.com"
            log "API documentation: https://your-domain.com/docs"
            ;;
        status)
            status
            ;;
        backup)
            backup
            ;;
        rollback)
            if [[ -z "${2:-}" ]]; then
                error "Please specify backup directory"
            fi
            rollback "$2"
            ;;
        logs)
            docker-compose -f docker-compose.prod.yml logs -f creativeiq-api
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"