# CreativeIQ Production Deployment Guide

Complete guide for deploying CreativeIQ to production with enterprise-grade security, monitoring, and scalability.

## 🚀 **Production Architecture**

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Load Balancer │    │    Nginx     │    │  CreativeIQ API │
│   (CloudFlare)  │◄──►│   (SSL/TLS)  │◄──►│  (Multi-worker) │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │                      │
                    ┌─────────┼──────────┬───────────┼───────────┐
                    │         │          │           │           │
            ┌───────▼───┐ ┌───▼────┐ ┌───▼───┐ ┌────▼────┐ ┌───▼────┐
            │PostgreSQL │ │MongoDB │ │ Redis │ │Training │ │Monitor │
            │(Analytics)│ │(Images)│ │(Cache)│ │ Worker  │ │ Stack  │
            └───────────┘ └────────┘ └───────┘ └─────────┘ └────────┘
```

## 📋 **Pre-Deployment Checklist**

### **Infrastructure Requirements**

**Minimum Server Specs:**
- **CPU**: 4 cores (8+ recommended for GPU)
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 50GB SSD (100GB+ recommended)
- **GPU**: Optional but recommended (NVIDIA with CUDA)
- **Network**: 1Gbps connection

**Supported Platforms:**
- ✅ Ubuntu 20.04/22.04 LTS
- ✅ CentOS 8/Rocky Linux 8
- ✅ Amazon Linux 2
- ✅ Docker/Kubernetes environments

### **Required Software**

```bash
# Install Docker & Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### **Domain & SSL Setup**

1. **Domain Configuration:**
   ```
   A     your-domain.com      → your-server-ip
   CNAME api.your-domain.com  → your-domain.com
   CNAME app.your-domain.com  → your-domain.com
   ```

2. **SSL Certificates:**
   ```bash
   # Let's Encrypt (recommended)
   sudo certbot certonly --standalone -d your-domain.com -d api.your-domain.com

   # Copy certificates
   sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/cert.pem
   sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/key.pem
   ```

## 🔧 **Environment Configuration**

### **Step 1: Clone and Setup**

```bash
# Clone repository
git clone https://github.com/your-org/CreativeIQ.git
cd CreativeIQ

# Setup production environment
make setup-prod
```

### **Step 2: Configure Environment Variables**

Edit `.env.prod` with your production settings:

```bash
# Critical security settings
SECRET_KEY=your-256-bit-secret-key-here
POSTGRES_PASSWORD=your-secure-db-password
MONGO_ROOT_PASSWORD=your-secure-mongo-password
REDIS_PASSWORD=your-secure-redis-password

# Domain configuration
ALLOWED_HOSTS=your-domain.com,api.your-domain.com
CORS_ORIGINS=https://your-domain.com,https://app.your-domain.com

# AI Model settings
HF_TOKEN=your_huggingface_token
DEVICE=cuda  # or 'cpu' if no GPU

# API Keys for data collection
INSTAGRAM_ACCESS_TOKEN=your_instagram_token
LINKEDIN_ACCESS_TOKEN=your_linkedin_token
FACEBOOK_ACCESS_TOKEN=your_facebook_token

# Monitoring
SENTRY_DSN=your_sentry_dsn_for_error_tracking
GRAFANA_PASSWORD=your_grafana_password
```

### **Step 3: SSL Certificate Setup**

```bash
# Setup SSL certificates
make setup-ssl

# Or manually copy your certificates
cp /path/to/your/cert.pem nginx/ssl/cert.pem
cp /path/to/your/key.pem nginx/ssl/key.pem
chmod 600 nginx/ssl/*
```

## 🚀 **Deployment Process**

### **Quick Deployment**

```bash
# One-command deployment
make deploy-prod
```

### **Manual Step-by-Step Deployment**

```bash
# 1. Environment check
./scripts/deploy.sh status

# 2. Create backup
make deploy-backup

# 3. Deploy
./scripts/deploy.sh deploy

# 4. Verify deployment
make health-check
```

### **Deployment with Model Training**

```bash
# Deploy and train initial models
TRAIN_INITIAL_MODELS=true make deploy-prod
```

## 📊 **Post-Deployment Configuration**

### **Database Setup**

```bash
# Run database migrations
make db-migrate

# Verify database
docker-compose -f docker-compose.prod.yml exec postgres psql -U creativeiq_prod -d creativeiq_prod -c "\dt"
```

### **Model Training Setup**

```bash
# Train production models with real data
make train-production

# Monitor training progress
make deploy-logs
```

### **SSL Certificate Renewal**

```bash
# Setup auto-renewal (crontab)
0 12 * * * /usr/bin/certbot renew --quiet
```

## 🔒 **Security Configuration**

### **Firewall Setup**

```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable

# Block direct access to database ports from outside
sudo ufw deny 5432  # PostgreSQL
sudo ufw deny 27017 # MongoDB
sudo ufw deny 6379  # Redis
```

### **Docker Security**

```bash
# Run containers as non-root
# Enable user namespaces
echo '{"userns-remap": "default"}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker
```

### **API Security**

- ✅ **Rate Limiting**: 10 requests/second
- ✅ **CORS Protection**: Domain-restricted
- ✅ **Request Size Limits**: 10MB max
- ✅ **HTTPS Enforcement**: All traffic redirected
- ✅ **Security Headers**: HSTS, CSP, XSS protection

## 📈 **Monitoring & Observability**

### **Monitoring Stack**

**Access Monitoring:**
- **Prometheus**: http://your-domain.com:9090
- **Grafana**: http://your-domain.com:3000
- **Application Logs**: `make monitor-logs`

### **Key Metrics to Monitor**

```yaml
# Application metrics
- API response time
- Analysis processing time
- Model accuracy
- Error rates
- Memory/CPU usage

# Business metrics
- Daily analyses count
- User engagement
- Platform performance
- Model prediction accuracy
```

### **Alerting Setup**

```bash
# Configure alerts in Grafana
# Email/Slack notifications for:
# - High error rates (>5%)
# - Slow response times (>30s)
# - System resource usage (>80%)
# - Model drift detection
```

## 🔄 **Maintenance & Updates**

### **Regular Backups**

```bash
# Daily database backup (crontab)
0 2 * * * cd /path/to/CreativeIQ && make db-backup

# Weekly full backup
0 3 * * 0 cd /path/to/CreativeIQ && make deploy-backup
```

### **Model Retraining**

```bash
# Weekly model retraining (crontab)
0 4 * * 0 cd /path/to/CreativeIQ && make retrain
```

### **Application Updates**

```bash
# Pull latest changes
git pull origin main

# Deploy updates
make deploy-prod

# Verify deployment
make health-check
```

### **Rollback Process**

```bash
# List available backups
ls backups/

# Rollback to previous version
./scripts/deploy.sh rollback backups/20240115_143000
```

## 🎯 **Performance Optimization**

### **GPU Configuration**

```bash
# Install NVIDIA Docker runtime
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2

# Enable GPU in production
echo 'DEVICE=cuda' >> .env.prod
```

### **Database Optimization**

```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET work_mem = '256MB';
SELECT pg_reload_conf();
```

### **Caching Strategy**

```bash
# Redis configuration for production
# Memory optimization
redis-cli CONFIG SET maxmemory 2gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

## 🌐 **Cloud Deployment Options**

### **AWS Deployment**

```bash
# ECS with GPU instances
# Recommended: p3.2xlarge for AI workloads
# Use Application Load Balancer
# RDS for PostgreSQL
# ElastiCache for Redis
```

### **Google Cloud Platform**

```bash
# Cloud Run with custom containers
# Cloud SQL for PostgreSQL
# Memorystore for Redis
# Vertex AI for model training
```

### **Kubernetes Deployment**

```yaml
# Use provided k8s manifests
kubectl apply -f k8s/
kubectl get pods -n creativeiq
```

## 🔍 **Troubleshooting**

### **Common Issues**

**API Not Responding:**
```bash
# Check container status
docker-compose -f docker-compose.prod.yml ps

# Check logs
make deploy-logs

# Restart services
docker-compose -f docker-compose.prod.yml restart
```

**Model Loading Failures:**
```bash
# Check GPU availability
nvidia-smi

# Check model cache
ls -la models/

# Download models manually
python -c "from transformers import LlavaNextProcessor; LlavaNextProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')"
```

**Database Connection Issues:**
```bash
# Test database connectivity
docker-compose -f docker-compose.prod.yml exec postgres pg_isready

# Check database logs
docker-compose -f docker-compose.prod.yml logs postgres
```

### **Performance Issues**

**Slow Analysis Times:**
- Enable GPU acceleration
- Increase worker processes
- Optimize model quantization
- Use Redis caching

**High Memory Usage:**
- Reduce concurrent analyses
- Enable model offloading
- Optimize image preprocessing
- Monitor memory leaks

## 📞 **Support & Maintenance**

### **Health Monitoring**

```bash
# Daily health checks
make health-check

# Weekly performance tests
make load-test

# Monthly security updates
sudo apt update && sudo apt upgrade
```

### **Log Management**

```bash
# View application logs
make monitor-logs

# Check error logs
tail -f logs/error.log

# Analyze performance logs
tail -f logs/performance.log
```

## 🎉 **Production Checklist**

- ✅ **Environment**: .env.prod configured
- ✅ **SSL**: Certificates installed and auto-renewal setup
- ✅ **Security**: Firewall, rate limiting, HTTPS enforcement
- ✅ **Database**: Migrations run, backups configured
- ✅ **Monitoring**: Prometheus/Grafana setup, alerts configured
- ✅ **Performance**: GPU enabled (if available), caching configured
- ✅ **Backups**: Daily database and weekly full backups
- ✅ **Updates**: Auto-updates configured for security patches
- ✅ **Documentation**: Team trained on deployment and maintenance

---

**🚀 Ready for Production!**

Your CreativeIQ platform is now enterprise-ready with:
- **High Availability**: Multi-container orchestration
- **Security**: HTTPS, rate limiting, security headers
- **Monitoring**: Real-time metrics and alerting
- **Scalability**: Horizontal and vertical scaling support
- **Reliability**: Automated backups and rollback capability

**Access your production deployment:**
- **Application**: https://your-domain.com
- **API Docs**: https://your-domain.com/docs
- **Monitoring**: https://your-domain.com:3000

**Need help?** Check the troubleshooting section or review application logs with `make monitor-logs`.