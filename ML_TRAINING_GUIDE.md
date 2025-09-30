# CreativeIQ ML Training Guide

Complete guide to training CreativeIQ's performance prediction models with real datasets.

## 🎯 **Understanding the ML Pipeline**

### **Current State vs Production**

**What we built (MVP):**
- ✅ Complete ML training infrastructure
- ✅ Dataset collection pipeline
- ✅ Model training and evaluation
- ✅ Production deployment system
- ⚠️ **Uses synthetic data for demo**

**For production with real data:**
- 📊 Need real social media API access
- 🎨 Need design performance datasets
- 👨‍🎨 Need expert design quality annotations

## 📊 **Dataset Requirements**

### **1. Social Media Performance Data**

**Instagram Dataset:**
```json
{
  "id": "post_123",
  "image_url": "https://...",
  "engagement_metrics": {
    "likes": 1250,
    "comments": 45,
    "shares": 12,
    "saves": 89,
    "reach": 15000,
    "impressions": 25000
  },
  "audience_size": 50000,
  "posting_time": "2024-01-15T14:30:00Z"
}
```

**LinkedIn Dataset:**
```json
{
  "id": "post_456",
  "image_url": "https://...",
  "engagement_metrics": {
    "likes": 342,
    "comments": 28,
    "shares": 67,
    "clicks": 890,
    "impressions": 12000,
    "engagement_rate": 0.045
  },
  "company_size": "1000-5000",
  "industry": "Technology"
}
```

### **2. Design Quality Dataset**

**Expert Annotations:**
```json
{
  "design_id": "design_789",
  "image_path": "/path/to/image.jpg",
  "expert_scores": {
    "overall_quality": 85,
    "color_harmony": 90,
    "typography": 80,
    "layout": 88,
    "brand_consistency": 75
  },
  "expert_id": "designer_001",
  "annotation_date": "2024-01-15"
}
```

### **3. A/B Testing Dataset**

**Test Results:**
```json
{
  "test_id": "ab_test_001",
  "variant_a": {
    "image_path": "/path/to/variant_a.jpg",
    "conversion_rate": 0.035,
    "click_through_rate": 0.12,
    "impressions": 10000
  },
  "variant_b": {
    "image_path": "/path/to/variant_b.jpg",
    "conversion_rate": 0.042,
    "click_through_rate": 0.15,
    "impressions": 10000
  },
  "statistical_significance": 0.95
}
```

## 🚀 **Training Pipeline Steps**

### **Step 1: Data Collection**

```bash
# Demo training (synthetic data)
make train-quick

# Production training (requires API keys)
make train-production
```

**Set up API access:**
```bash
# Add to .env file
INSTAGRAM_ACCESS_TOKEN=your_token
LINKEDIN_ACCESS_TOKEN=your_token
FACEBOOK_ACCESS_TOKEN=your_token
DRIBBBLE_ACCESS_TOKEN=your_token
```

### **Step 2: Dataset Validation**

The pipeline automatically validates:
- ✅ Minimum sample requirements (100+ samples)
- ✅ Platform distribution balance
- ✅ Missing data analysis
- ✅ Image file integrity
- ✅ Data quality scoring

### **Step 3: Feature Extraction**

**Visual Features Extracted:**
- **Color:** Palette, harmony, accessibility scores
- **Typography:** Font pairing, readability, hierarchy
- **Layout:** Composition, balance, grid alignment
- **Context:** Platform, timing, audience size

**Example feature vector:**
```python
{
  "color_harmony_score": 85.2,
  "readability_score": 92.1,
  "composition_score": 78.5,
  "aspect_ratio": 1.0,
  "platform_instagram": 1,
  "posting_hour": 14,
  "audience_size": 50000
  # ... 20+ features total
}
```

### **Step 4: Model Training**

**Models Trained:**
- 🌲 **Random Forest** (ensemble, good baseline)
- 🚀 **XGBoost** (gradient boosting, often best)
- 🧠 **Neural Network** (complex patterns)
- 📈 **Gradient Boosting** (iterative improvement)

**Platform-Specific Models:**
- `general_model.joblib` - All platforms
- `instagram_model.joblib` - Instagram-specific
- `linkedin_model.joblib` - LinkedIn-specific
- `facebook_model.joblib` - Facebook-specific

### **Step 5: Model Evaluation**

**Metrics Tracked:**
- **R² Score:** How well model explains variance (target: >0.7)
- **RMSE:** Root mean squared error (lower is better)
- **Cross-validation:** 5-fold validation for robustness
- **Feature importance:** Which features matter most

### **Step 6: Production Deployment**

Models automatically deployed to `models/trained/` and loaded by the production predictor.

## 🧪 **Running Training**

### **Quick Demo (5 minutes)**

```bash
# Shows concept with synthetic data
python scripts/demo_training.py --mode quick
```

**Output:**
```
⚡ Quick ML Demo
🧠 Training concept:
1. Collect design images + engagement metrics
2. Extract visual features (color, layout, typography)
3. Train ML models to predict engagement
4. Deploy models for real-time predictions

✅ Model trained!
📈 Sample predictions: [78.2 65.4 91.1]
🎯 Actual scores: [75.1 68.9 89.7]
```

### **Full Demo (20 minutes)**

```bash
# Complete pipeline with synthetic dataset
python scripts/demo_training.py --mode full
```

**Creates:**
- 200 synthetic design samples
- Trains 4 different ML models
- Evaluates performance
- Deploys best models

### **Production Training**

```bash
# With real API data
python scripts/train_models.py --mode full --collect-data --platforms instagram linkedin
```

## 📋 **Required API Setup**

### **Instagram Basic Display API**

1. Create Facebook Developer account
2. Create app and add Instagram Basic Display
3. Get access token with `user_media` scope

```python
# In dataset_collector.py
instagram_data = await collector.collect_instagram_data(
    user_ids=["design_account_1", "design_account_2"],
    days_back=30
)
```

### **LinkedIn Marketing API**

1. Create LinkedIn Developer account
2. Apply for Marketing API access
3. Get access token with analytics permissions

### **A/B Testing Integration**

**Optimizely Integration:**
```python
# Custom A/B test data collector
ab_test_data = await collector.collect_a_b_test_data([
    {"test_id": "color_test", "platform": "instagram"},
    {"test_id": "layout_test", "platform": "linkedin"}
])
```

## 📊 **Model Performance Targets**

### **Production Benchmarks**

| Metric | Target | Current Demo |
|--------|---------|--------------|
| R² Score | >0.70 | ~0.65 (synthetic) |
| RMSE | <15 | ~18 (synthetic) |
| Prediction Time | <500ms | ~200ms |
| Feature Count | 25-30 | 20 |

### **Platform-Specific Performance**

| Platform | Data Requirement | Model Focus |
|----------|------------------|-------------|
| Instagram | Visual-heavy content | Color + Composition |
| LinkedIn | Professional designs | Typography + Accessibility |
| TikTok | Mobile-first content | Aspect ratio + Complexity |
| Facebook | Mixed content | Balanced features |

## 🔄 **Continuous Learning**

### **Incremental Training**

```bash
# Retrain with new data weekly
make retrain
```

**Process:**
1. Collect past 7 days of data
2. Validate data quality
3. Retrain existing models
4. A/B test new vs old models
5. Deploy if performance improves

### **Model Monitoring**

**Track in production:**
- Prediction accuracy vs actual results
- Model drift over time
- Feature importance changes
- Platform performance trends

## 🎯 **Getting Real Data**

### **Social Media APIs**

**Instagram Graph API:**
```python
# Business accounts only
url = f"https://graph.facebook.com/v18.0/{account_id}/media"
params = {
    "fields": "id,media_type,media_url,like_count,comments_count,timestamp",
    "access_token": access_token
}
```

**LinkedIn Marketing API:**
```python
# Company page analytics
url = f"https://api.linkedin.com/v2/shares"
headers = {"Authorization": f"Bearer {access_token}"}
```

### **Design Platform APIs**

**Dribbble API:**
```python
# Popular shots with engagement data
url = "https://api.dribbble.com/v2/shots"
params = {
    "sort": "popular",
    "timeframe": "month",
    "access_token": dribbble_token
}
```

**Behance API:**
```python
# Creative work with stats
url = "https://www.behance.net/v2/projects"
params = {
    "api_key": behance_key,
    "sort": "appreciations"
}
```

## 💡 **Advanced Features**

### **Brand Consistency Learning**

```python
# Train brand-specific models
brand_model = await trainer.train_brand_model(
    brand_guidelines=brand_assets,
    sample_designs=brand_portfolio
)
```

### **Trend Detection**

```python
# Identify trending design patterns
trends = await analyzer.detect_trends(
    time_window_days=30,
    minimum_samples=50
)
```

### **Competitive Analysis**

```python
# Compare against competitor designs
competitor_analysis = await analyzer.benchmark_against_competitors(
    your_design=image,
    competitor_accounts=["competitor1", "competitor2"]
)
```

## 🚀 **Next Steps**

1. **Start with Demo:** Run `make train-quick` to understand the pipeline
2. **Get API Access:** Set up social media API credentials
3. **Collect Real Data:** Use the data collection pipeline
4. **Train Production Models:** Run with real engagement data
5. **Monitor & Improve:** Set up continuous learning pipeline

---

**Ready to train your first model?**

```bash
# Quick start
make train-quick

# See training options
python scripts/train_models.py --help
```