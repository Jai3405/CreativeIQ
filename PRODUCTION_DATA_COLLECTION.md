# CreativeIQ Production Data Collection Guide

**Real Social Media Data Integration for AI Training**

CreativeIQ now supports collecting real data from Instagram, LinkedIn, and Facebook APIs instead of synthetic demo data. This guide shows you how to set up and use production data collection.

## 🚀 **Quick Start**

```bash
# 1. Check API status
make check-api-status

# 2. Collect data from Instagram (authenticated user)
make collect-instagram

# 3. Train models with real data
make train-production

# 4. Deploy to production
make deploy-prod
```

## 🔑 **API Setup Required**

### **Instagram Basic Display API**
```bash
# Get access token at: https://developers.facebook.com/docs/instagram-basic-display-api
# Required permissions: user_profile, user_media
INSTAGRAM_ACCESS_TOKEN=your_instagram_token
```

### **LinkedIn Marketing API**
```bash
# Get access token at: https://developer.linkedin.com/
# Required permissions: r_liteprofile, r_emailaddress, r_ads
LINKEDIN_ACCESS_TOKEN=your_linkedin_token
```

### **Facebook Graph API**
```bash
# Get access token at: https://developers.facebook.com/tools/explorer/
# Required permissions: pages_read_engagement, pages_read_user_content
FACEBOOK_ACCESS_TOKEN=your_facebook_token
```

Add these to your `.env` file for development or `.env.prod` for production.

## 📊 **Data Collection Commands**

### **Check API Status**
```bash
# Verify all API credentials
make check-api-status

# Or directly
python scripts/collect_production_data.py --check-apis
```

### **Collect Instagram Data**
```bash
# Collect from authenticated user's account
make collect-instagram

# Collect from specific business accounts
python scripts/collect_production_data.py --instagram-business account_id_1 account_id_2

# Collect from specific users (requires user permissions)
python scripts/collect_production_data.py --instagram-users user_id_1 user_id_2
```

### **Collect LinkedIn Data**
```bash
# Collect from company pages
python scripts/collect_production_data.py --linkedin-companies company_id_1 company_id_2

# Collect from user profiles
python scripts/collect_production_data.py --linkedin-users user_id_1 user_id_2
```

### **Collect Facebook Data**
```bash
# Collect from Facebook pages
python scripts/collect_production_data.py --facebook-pages page_id_1 page_id_2

# Collect ad creative performance
python scripts/collect_production_data.py --facebook-ads ad_account_id_1
```

### **Collect from All Platforms**
```bash
# Comprehensive data collection
make collect-all-platforms

# Or with custom configuration
python scripts/collect_production_data.py \
  --instagram-business me \
  --linkedin-companies your_company_id \
  --facebook-pages your_page_id \
  --include-design-platforms
```

## 🎯 **Training with Real Data**

### **Full Production Training**
```bash
# Train models with real API data collection
python scripts/train_models.py --mode full --collect-data --platforms instagram linkedin facebook

# Or use Makefile
make train-production
```

### **Quick Training (Development)**
```bash
# Quick training with existing data
python scripts/train_models.py --mode quick

# Or use Makefile
make train-models
```

### **Incremental Retraining**
```bash
# Retrain with new data (for production updates)
python scripts/train_models.py --mode incremental

# Or use Makefile
make retrain
```

## 📈 **Data Collection Results**

### **What Gets Collected**

**Instagram Data:**
- ✅ User posts with engagement metrics (likes, comments, shares)
- ✅ Business account insights (reach, impressions, saves)
- ✅ Hashtag performance analysis
- ✅ Image files automatically downloaded
- ✅ Posting time analysis

**LinkedIn Data:**
- ✅ Company page posts with professional engagement
- ✅ User posts and articles
- ✅ Ad creative performance data
- ✅ Industry and company context
- ✅ Click-through and conversion metrics

**Facebook Data:**
- ✅ Page posts with detailed engagement
- ✅ Ad creative insights and performance
- ✅ Audience demographics and reach
- ✅ Hashtag content discovery
- ✅ Cross-platform engagement comparison

### **Data Storage Structure**
```
data/training/
├── images/                    # Downloaded social media images
│   ├── instagram_*.jpg
│   ├── linkedin_*.jpg
│   └── facebook_*.jpg
├── metadata/                  # Engagement and performance data
│   ├── instagram_production.json
│   ├── linkedin_production.json
│   ├── facebook_production.json
│   └── production_training_dataset.json
└── models/                    # Trained ML models
    ├── instagram_predictor.pkl
    ├── linkedin_predictor.pkl
    └── general_predictor.pkl
```

## 🔧 **Configuration**

### **Environment Variables (.env)**
```bash
# Required for production data collection
INSTAGRAM_ACCESS_TOKEN=your_instagram_token
LINKEDIN_ACCESS_TOKEN=your_linkedin_token
FACEBOOK_ACCESS_TOKEN=your_facebook_token

# Optional - AI model configuration
HF_TOKEN=your_huggingface_token
DEVICE=cuda  # or 'cpu'

# Database (for production)
POSTGRES_PASSWORD=secure_password
MONGO_ROOT_PASSWORD=secure_password
REDIS_PASSWORD=secure_password
```

### **Account Configuration**

Edit `scripts/train_models.py` to configure your specific accounts:

```python
def get_instagram_business_accounts():
    return ["me"]  # Use authenticated user, or add specific account IDs

def get_linkedin_companies():
    return ["your_company_id"]  # Add your LinkedIn company IDs

def get_facebook_pages():
    return ["your_page_id"]  # Add your Facebook page IDs
```

## 🚀 **Production Deployment**

### **1. Setup Environment**
```bash
# Copy production environment template
cp .env.prod .env.prod.local

# Edit with your API credentials
nano .env.prod.local
```

### **2. Collect Production Data**
```bash
# Collect real data for training
make collect-all-platforms
```

### **3. Train Production Models**
```bash
# Train with real collected data
make train-production
```

### **4. Deploy to Production**
```bash
# Deploy with trained models
make deploy-prod
```

## 📊 **Monitoring and Analytics**

### **Data Quality Metrics**
- **Sample Count:** Minimum 100 samples per platform
- **Engagement Coverage:** >80% of posts have engagement data
- **Image Quality:** All images >100x100 pixels
- **Date Range:** Training data from last 30 days
- **API Health:** All configured APIs responding correctly

### **Model Performance Tracking**
- **R² Score:** >0.85 for platform-specific models
- **RMSE:** <0.15 for engagement predictions
- **Cross-Validation:** 5-fold validation on all models
- **Feature Importance:** Color, typography, layout analysis

### **Production Monitoring**
```bash
# Check system health
make health-check

# Monitor training metrics
make monitor-metrics

# View collection logs
make monitor-logs
```

## 🔒 **Security and Privacy**

### **API Token Security**
- ✅ Store tokens in environment variables only
- ✅ Never commit tokens to version control
- ✅ Use least-privilege permissions
- ✅ Rotate tokens regularly
- ✅ Monitor token usage and expiration

### **Data Privacy**
- ✅ Only collect public posts and authorized content
- ✅ Respect platform rate limits
- ✅ Store data securely with encryption
- ✅ Implement data retention policies
- ✅ Comply with platform Terms of Service

### **Rate Limiting**
- **Instagram:** 200 requests/hour per token
- **LinkedIn:** 500 requests/day per token
- **Facebook:** 200 requests/hour per app

Built-in rate limiting ensures compliance with all platform limits.

## 🛠️ **Troubleshooting**

### **No Data Collected**
```bash
# Check API credentials
make check-api-status

# Verify token permissions
python scripts/collect_production_data.py --setup-guide

# Test individual platform
python scripts/collect_production_data.py --instagram-business me --check-apis
```

### **API Token Issues**
- **Invalid Token:** Regenerate token with proper permissions
- **Expired Token:** Get new token from platform developer console
- **Rate Limited:** Wait for rate limit reset or reduce collection frequency

### **Low Data Quality**
- **Few Samples:** Expand account list or increase collection timeframe
- **Missing Images:** Check account privacy settings and image URLs
- **Poor Engagement:** Focus on accounts with regular posting and engagement

## 📞 **Support**

### **Get Help**
```bash
# Show API setup guide
make setup-api-guide

# Check comprehensive troubleshooting
python scripts/collect_production_data.py --setup-guide
```

### **Platform Documentation**
- [Instagram Basic Display API](https://developers.facebook.com/docs/instagram-basic-display-api)
- [LinkedIn Marketing API](https://docs.microsoft.com/en-us/linkedin/marketing/)
- [Facebook Graph API](https://developers.facebook.com/docs/graph-api/)

---

## 🎉 **Success Criteria**

Your production data collection is successful when:

✅ **API Status:** All configured APIs return valid responses
✅ **Data Volume:** >500 samples collected across platforms
✅ **Data Quality:** >90% of images downloaded successfully
✅ **Model Training:** R² score >0.85 for engagement prediction
✅ **Production Ready:** Models deployed and responding to requests

**Ready to go live!** 🚀

Your CreativeIQ platform now uses real social media data for AI-powered design intelligence, providing professional-grade insights based on actual performance metrics from Instagram, LinkedIn, and Facebook.