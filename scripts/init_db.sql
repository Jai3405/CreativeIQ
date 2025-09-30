-- CreativeIQ Database Initialization Script

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS models;

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id VARCHAR(255) UNIQUE NOT NULL,
    image_path TEXT NOT NULL,
    platform VARCHAR(50),
    status VARCHAR(20) DEFAULT 'pending',
    overall_score DECIMAL(5,2),
    color_harmony_score DECIMAL(5,2),
    typography_score DECIMAL(5,2),
    layout_score DECIMAL(5,2),
    hierarchy_score DECIMAL(5,2),
    performance_score DECIMAL(5,2),
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- User sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_agent TEXT,
    ip_address INET,
    analyses_count INTEGER DEFAULT 0,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Training datasets table
CREATE TABLE IF NOT EXISTS training_datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_name VARCHAR(255) NOT NULL,
    platform VARCHAR(50),
    sample_count INTEGER,
    collection_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    data_quality_score DECIMAL(5,2),
    metadata JSONB,
    file_path TEXT
);

-- Model training history
CREATE TABLE IF NOT EXISTS model_training_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    platform VARCHAR(50),
    training_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    training_end TIMESTAMP WITH TIME ZONE,
    r2_score DECIMAL(5,4),
    rmse DECIMAL(8,4),
    training_samples INTEGER,
    model_path TEXT,
    hyperparameters JSONB,
    feature_importance JSONB,
    status VARCHAR(20) DEFAULT 'training'
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    platform VARCHAR(50),
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tags JSONB
);

-- Feedback table
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id VARCHAR(255) REFERENCES analysis_results(analysis_id),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    comments TEXT,
    feedback_type VARCHAR(50) DEFAULT 'quality',
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET
);

-- A/B testing results
CREATE TABLE IF NOT EXISTS ab_test_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    test_id VARCHAR(255) NOT NULL,
    variant_name VARCHAR(100) NOT NULL,
    platform VARCHAR(50),
    image_path TEXT,
    conversion_rate DECIMAL(6,4),
    click_through_rate DECIMAL(6,4),
    engagement_rate DECIMAL(6,4),
    impressions INTEGER,
    clicks INTEGER,
    conversions INTEGER,
    statistical_significance DECIMAL(4,3),
    test_start TIMESTAMP WITH TIME ZONE,
    test_end TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_analysis_results_created_at ON analysis_results(created_at);
CREATE INDEX IF NOT EXISTS idx_analysis_results_platform ON analysis_results(platform);
CREATE INDEX IF NOT EXISTS idx_analysis_results_status ON analysis_results(status);
CREATE INDEX IF NOT EXISTS idx_user_sessions_last_seen ON user_sessions(last_seen);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_recorded_at ON performance_metrics(recorded_at);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_metric_name ON performance_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_training_history_platform ON model_training_history(platform);
CREATE INDEX IF NOT EXISTS idx_ab_test_results_test_id ON ab_test_results(test_id);

-- Views for analytics
CREATE OR REPLACE VIEW analytics.daily_analysis_stats AS
SELECT
    DATE(created_at) as analysis_date,
    platform,
    COUNT(*) as total_analyses,
    AVG(overall_score) as avg_score,
    AVG(processing_time_ms) as avg_processing_time,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_analyses,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_analyses
FROM analysis_results
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at), platform
ORDER BY analysis_date DESC, platform;

CREATE OR REPLACE VIEW analytics.platform_performance AS
SELECT
    platform,
    COUNT(*) as total_analyses,
    AVG(overall_score) as avg_overall_score,
    AVG(color_harmony_score) as avg_color_score,
    AVG(typography_score) as avg_typography_score,
    AVG(layout_score) as avg_layout_score,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY overall_score) as median_score,
    STDDEV(overall_score) as score_stddev
FROM analysis_results
WHERE status = 'completed'
    AND created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY platform;

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_analysis_results_updated_at
    BEFORE UPDATE ON analysis_results
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT USAGE ON SCHEMA analytics TO creativeiq_prod;
GRANT USAGE ON SCHEMA models TO creativeiq_prod;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO creativeiq_prod;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO creativeiq_prod;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO creativeiq_prod;

-- Insert initial data
INSERT INTO performance_metrics (metric_name, metric_value, platform, tags) VALUES
('model_accuracy', 0.85, 'general', '{"model_type": "random_forest"}'),
('avg_processing_time', 2500, 'general', '{"unit": "milliseconds"}'),
('daily_analyses', 0, 'general', '{"date": "today"}')
ON CONFLICT DO NOTHING;