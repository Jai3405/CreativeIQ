# CreativeIQ Quick Start Guide

Welcome to CreativeIQ! This guide will get you up and running in minutes.

## 🚀 Choose Your Setup Method

### Option A: Docker (Recommended for Production)

1. **Prerequisites**: Docker and Docker Compose installed
2. **Quick Start**:
   ```bash
   # Clone and enter directory
   cd CreativeIQ

   # Start all services
   docker-compose up -d

   # Check status
   docker-compose ps
   ```

3. **Access**:
   - Web Interface: http://localhost
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Option B: Local Development

1. **Prerequisites**: Python 3.11+
2. **Quick Setup**:
   ```bash
   # Run setup script
   python scripts/setup.py

   # Or use Makefile
   make setup
   ```

3. **Start Server**:
   ```bash
   # Start API server
   uvicorn app.main:app --reload

   # Or use Makefile
   make run
   ```

4. **Serve Frontend**:
   ```bash
   # In another terminal
   cd frontend
   python -m http.server 3000
   ```

## ⚡ Quick Test

1. **Upload a Design**: Go to http://localhost (or localhost:3000 for local)
2. **Drag & Drop**: Any image file (JPG, PNG, WebP)
3. **Analyze**: Click "Analyze Design"
4. **Get Results**: View scores, recommendations, and insights

## 🔧 Configuration

Edit `.env` file for customization:

```env
# AI Model (change if you have GPU)
DEVICE=cuda  # or 'cpu'
MODEL_NAME=llava-hf/llava-1.5-7b-hf

# Add HuggingFace token for private models
HF_TOKEN=your_token_here

# Database connections (if using external databases)
POSTGRES_SERVER=your_postgres_host
MONGODB_URL=mongodb://your_mongo_host:27017
```

## 📊 API Usage Examples

### Analyze Image
```bash
curl -X POST "http://localhost:8000/api/v1/analyze/" \
  -F "file=@your-design.jpg" \
  -F "target_platform=instagram"
```

### Chat with AI Coach
```bash
curl -X POST "http://localhost:8000/api/v1/chat/" \
  -H "Content-Type: application/json" \
  -d '{"message": "How can I improve my color scheme?"}'
```

### Check Performance
```bash
curl "http://localhost:8000/health"
```

## 🧪 Run Tests

```bash
# Using scripts
python scripts/run_tests.py

# Using Makefile
make test

# Using pytest directly
pytest tests/ -v
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Run `pip install -r requirements.txt`
2. **GPU Not Detected**: Install CUDA toolkit, set `DEVICE=cuda` in .env
3. **Port Already in Use**: Change ports in docker-compose.yml
4. **Large Model Download**: First run downloads ~7GB model - be patient!

### Check System Status

```bash
# Health check
curl http://localhost:8000/health

# Model status
curl http://localhost:8000/api/v1/health/models

# Docker logs
docker-compose logs creativeiq-api
```

### Performance Tips

- **GPU**: Use CUDA for 3-5x faster analysis
- **Memory**: 8GB+ RAM recommended
- **Storage**: Models need ~10GB space
- **Network**: Fast internet for initial model download

## 🎯 What's Next?

1. **Test with Your Designs**: Upload real projects
2. **Explore API**: Visit http://localhost:8000/docs
3. **Customize Analysis**: Modify prompts and parameters
4. **Integrate**: Use API in your existing workflows
5. **Scale**: Deploy to cloud with GPU instances

## 📖 Documentation

- **Full README**: See README.md for complete documentation
- **API Docs**: http://localhost:8000/docs (when running)
- **Architecture**: See technical details in README.md

## 🤝 Getting Help

- **Issues**: Report bugs via GitHub Issues
- **Questions**: Check API documentation first
- **Performance**: Monitor with `make monitor`

---

**You're all set! Start analyzing designs and getting AI-powered insights! 🎨**