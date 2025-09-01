"""
Main Streamlit Application for Market Trend Analysis
Provides a user-friendly interface for market analysis and predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf


from data_collection import MarketDataCollector
from preprocessing import DataPreprocessor
from analysis import MarketAnalyzer
from visualization import MarketVisualizer
from model import MarketPredictor

# Page configuration
st.set_page_config(
    page_title="Market Trend Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .ml-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .model-performance {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

def generate_investment_recommendation(data, current_price, price_change, ma_20, ma_50):
    """Generate investment recommendation based on technical analysis"""
    
    # Calculate additional indicators
    rsi = calculate_rsi(data['Close']) if len(data) >= 14 else 50
    volatility = data['Close'].pct_change().std() * np.sqrt(252) if len(data) > 1 else 0.2
    
    # Trend analysis
    price_above_ma20 = current_price > ma_20
    price_above_ma50 = current_price > ma_50
    ma_trend = ma_20 > ma_50
    
    # Momentum analysis
    recent_momentum = data['Close'].pct_change(periods=5).iloc[-1] if len(data) >= 6 else 0
    
    # Decision logic
    bullish_signals = 0
    bearish_signals = 0
    
    # Price vs Moving Averages
    if price_above_ma20:
        bullish_signals += 1
    else:
        bearish_signals += 1
        
    if price_above_ma50:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # Moving Average Trend
    if ma_trend:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # RSI Analysis
    if rsi < 30:
        bullish_signals += 2  # Oversold
    elif rsi > 70:
        bearish_signals += 2  # Overbought
    elif 40 < rsi < 60:
        bullish_signals += 0.5  # Neutral
    
    # Momentum
    if recent_momentum > 0:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # Price Change
    if price_change > 0:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # Volatility adjustment
    if volatility > 0.3:  # High volatility
        bullish_signals *= 0.8
        bearish_signals *= 0.8
    
    # Generate recommendation
    if bullish_signals > bearish_signals + 1:
        action = "BUY"
        confidence = min(90, 50 + (bullish_signals - bearish_signals) * 10)
        risk_level = "Low" if volatility < 0.2 else "Medium"
        reasoning = f"Strong bullish signals: Price above moving averages, positive momentum, RSI at {rsi:.1f}"
    elif bearish_signals > bullish_signals + 1:
        action = "SELL"
        confidence = min(90, 50 + (bearish_signals - bullish_signals) * 10)
        risk_level = "High" if volatility > 0.3 else "Medium"
        reasoning = f"Strong bearish signals: Price below moving averages, negative momentum, RSI at {rsi:.1f}"
    else:
        action = "HOLD"
        confidence = 60
        risk_level = "Medium"
        reasoning = f"Mixed signals: Price near moving averages, RSI at {rsi:.1f}, momentum neutral"
    
    return {
        'action': action,
        'confidence': confidence,
        'risk_level': risk_level,
        'reasoning': reasoning
    }

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50

def main():
    """Main application function"""
    
    # Initialize components at the top level
    collector = MarketDataCollector()
    preprocessor = DataPreprocessor()
    analyzer = MarketAnalyzer()
    visualizer = MarketVisualizer()
    predictor = MarketPredictor()
    
    # Header
    st.markdown('<h1 class="main-header">📈 Market Trend Analysis</h1>', unsafe_allow_html=True)
    
    # Check if we have cached data
    if 'processed_data' in st.session_state:
        st.subheader("📊 **Data Status**")
        st.info(f"✅ Cached data available for {st.session_state.get('ml_symbol', 'Unknown')} ({st.session_state['processed_data'].shape[0]} rows)")
        st.caption("This data will be used for machine learning analysis and can be cleared using the button below.")
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Add subheading at top of sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 **Market Analysis Dashboard**")
    st.sidebar.markdown("Configure your analysis parameters below:")
    st.sidebar.markdown("---")
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.selectbox(
        "Select data source:",
        ["Yahoo Finance", "Sample Data"]
    )
    
    # Symbol input
    symbol = st.sidebar.text_input("Enter Stock Symbol:", value="AAPL").upper()
    
    # Time period selection
    period = st.sidebar.selectbox(
        "Select time period:",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    # Interval selection
    interval = st.sidebar.selectbox(
        "Select interval:",
        ["1d", "1wk", "1mo"],
        index=0
    )
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    show_technical_indicators = st.sidebar.checkbox("Show Technical Indicators", value=True)
    show_volume = st.sidebar.checkbox("Show Volume", value=True)
    enable_ml = st.sidebar.checkbox("Enable Machine Learning", value=True)
    
    # Machine Learning options
    if enable_ml:
        st.sidebar.subheader("ML Configuration")
        ml_task = st.sidebar.selectbox(
            "ML Task:",
            ["Price Prediction", "Trend Classification", "Volatility Prediction", "Risk Assessment"]
        )
        
        # Dynamic model options based on ML Task
        if ml_task in ["Price Prediction", "Volatility Prediction"]:
            # Regression tasks
            ml_model_options = ["Random Forest", "Linear Regression", "SVR", "Neural Network"]
        else:
            # Classification tasks
            ml_model_options = ["Random Forest", "Logistic Regression", "SVC", "Neural Network"]
        
        ml_model = st.sidebar.selectbox(
            "ML Model:",
            ml_model_options
        )
        
        prediction_horizon = st.sidebar.slider(
            "Prediction Horizon (days):",
            min_value=1,
            max_value=30,
            value=5
        )
    
    # Main content area
    if st.sidebar.button("🚀 Analyze Market", type="primary", key="analyze_market_main"):
        with st.spinner("Fetching and analyzing market data..."):
            try:
                # Fetch data
                if data_source == "Yahoo Finance":
                    st.info(f"Fetching data for {symbol} from Yahoo Finance...")
                    data = collector.get_stock_data(symbol, period, interval)
                else:
                    # Generate sample data
                    st.info("Generating sample data for demonstration...")
                    dates = pd.date_range(
                        start=datetime.now() - timedelta(days=365),
                        end=datetime.now(),
                        freq='D'
                    )
                    np.random.seed(42)
                    data = pd.DataFrame({
                        'Date': dates,
                        'Open': np.random.randn(len(dates)).cumsum() + 100,
                        'High': np.random.randn(len(dates)).cumsum() + 102,
                        'Low': np.random.randn(len(dates)).cumsum() + 98,
                        'Close': np.random.randn(len(dates)).cumsum() + 100,
                        'Volume': np.random.randint(1000000, 10000000, len(dates)),
                        'Symbol': [symbol] * len(dates)
                    })
                
                if data.empty:
                    st.error("No data retrieved. Please check the symbol and try again.")
                    return
                
                # Data preprocessing
                st.success("Data fetched successfully! Processing...")
                cleaned_data = preprocessor.clean_data(data)
                st.info(f"After cleaning: {cleaned_data.shape} rows, {len(cleaned_data.columns)} columns")
                st.info(f"Columns after cleaning: {list(cleaned_data.columns)}")
                
                if show_technical_indicators:
                    st.info("Adding technical indicators...")
                    data_with_indicators = preprocessor.add_technical_indicators(cleaned_data)
                    st.info(f"After indicators: {data_with_indicators.shape} rows, {len(data_with_indicators.columns)} columns")
                    
                    # Check for Price_Change specifically after indicators
                    if 'Price_Change' in data_with_indicators.columns:
                        st.success("✅ Price_Change found after adding indicators!")
                    else:
                        st.error("❌ Price_Change still missing after indicators!")
                        st.info(f"Columns after indicators: {list(data_with_indicators.columns)}")
                    
                    st.info("Creating additional features...")
                    data_with_features = preprocessor.create_features(data_with_indicators)
                    st.info(f"After features: {data_with_features.shape} rows, {len(data_with_features.columns)} columns")
                else:
                    st.info("Skipping technical indicators")
                    data_with_features = cleaned_data
                
                # Debug: Show available columns
                st.info(f"Available columns after preprocessing: {list(data_with_features.columns)}")
                
                # Check for Price_Change specifically
                if 'Price_Change' in data_with_features.columns:
                    st.success("✅ Price_Change column found!")
                else:
                    st.error("❌ Price_Change column missing!")
                    
                    # Try to manually create Price_Change
                    st.info("Attempting to manually create Price_Change...")
                    try:
                        data_with_features['Price_Change'] = data_with_features['Close'].pct_change()
                        data_with_features['Price_Change_5'] = data_with_features['Close'].pct_change(periods=5)
                        data_with_features['Price_Change_20'] = data_with_features['Close'].pct_change(periods=20)
                        st.success("✅ Price_Change columns manually created!")
                    except Exception as e:
                        st.error(f"❌ Failed to manually create Price_Change: {str(e)}")
                    
                    st.info("Checking what price-related columns exist:")
                    price_cols = [col for col in data_with_features.columns if 'price' in col.lower() or 'change' in col.lower()]
                    st.info(f"Price-related columns: {price_cols}")
                
                # Market analysis
                st.success("Performing market analysis...")
                insights = analyzer.generate_market_insights(data_with_features)
                
                # Display results
                display_results(data_with_features, insights, visualizer, symbol)
                
                # Enhanced Machine Learning section
                if enable_ml:
                    # Store data in session state for persistence
                    st.session_state['processed_data'] = data_with_features
                    st.session_state['ml_insights'] = insights
                    st.session_state['ml_symbol'] = symbol
                    st.session_state['ml_task'] = ml_task # Store ml_task
                    st.session_state['ml_model'] = ml_model # Store ml_model
                    
                    display_enhanced_ml_section(data_with_features, predictor, ml_task, ml_model, prediction_horizon)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
    
    # Always show ML section if enabled and we have cached data
    if enable_ml and 'processed_data' in st.session_state:
        st.markdown("---")
        # Pass the current sidebar values to keep them in sync
        display_enhanced_ml_section(
            st.session_state['processed_data'], 
            predictor, 
            ml_task,  # Use current sidebar value
            ml_model,  # Use current sidebar value
            prediction_horizon
        )

def display_results(data, insights, visualizer, symbol):
    """Display analysis results"""
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Price Chart", 
        "📈 Trend Analysis", 
        "📉 Volatility Analysis", 
        "🔍 Market Insights", 
        "📋 Summary Dashboard"
    ])
    
    with tab1:
        st.subheader(f"{symbol} Price Chart")
        
        # Chart options
        col1, col2 = st.columns(2)
        with col1:
            show_indicators = st.checkbox("Show Technical Indicators", value=True, key="show_indicators_chart")
        with col2:
            show_volume = st.checkbox("Show Volume", value=True, key="show_volume_chart")
        
        # Create price chart
        price_chart = visualizer.create_price_chart(
            data, symbol, show_volume=show_volume, show_indicators=show_indicators
        )
        st.plotly_chart(price_chart, use_container_width=True, key="price_chart")

    
    with tab2:
        st.subheader("Trend Analysis")
        
        if insights and 'trend_analysis' in insights:
            trend_data = insights['trend_analysis']
            
            # Display trend metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Trend Direction", trend_data.get('trend_direction', 'N/A'))
            with col2:
                st.metric("Trend Strength", f"{trend_data.get('trend_strength', 0):.2f}")
            with col3:
                st.metric("Current MA Trend", trend_data.get('current_ma_trend', 'N/A'))
            with col4:
                st.metric("5-Day Momentum", f"{trend_data.get('momentum_5d', 0):.2f}%")
            
            # Trend analysis chart
            trend_chart = visualizer.create_trend_analysis_chart(data, insights)
            st.plotly_chart(trend_chart, use_container_width=True,key="trend_chart")
        else:
            st.warning("Trend analysis data not available")
    
    with tab3:
        st.subheader("Volatility Analysis")
        
        if insights and 'volatility_analysis' in insights:
            vol_data = insights['volatility_analysis']
            
            # Display volatility metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Daily Volatility", f"{vol_data.get('daily_volatility', 0):.2%}")
            with col2:
                st.metric("Max Drawdown", f"{vol_data.get('max_drawdown', 0):.2%}")
            with col3:
                st.metric("VaR 95%", f"{vol_data.get('var_95', 0):.2%}")
            with col4:
                st.metric("Volatility Regime", vol_data.get('volatility_regime', 'N/A'))
            
            # Volatility chart
            vol_chart = visualizer.create_volatility_chart(data, vol_data)
            st.plotly_chart(vol_chart, use_container_width=True,key="vol_chart")
        else:
            st.warning("Volatility analysis data not available")
    
    with tab4:
        st.subheader("Market Insights")
        
        if insights and 'summary' in insights:
            summary = insights['summary']
            
            # Market sentiment
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment = summary.get('overall_market_sentiment', 'Neutral')
                sentiment_color = {
                    'Bullish': '🟢',
                    'Bearish': '🔴',
                    'Neutral': '🟡'
                }.get(sentiment, '⚪')
                st.metric("Market Sentiment", f"{sentiment_color} {sentiment}")
            
            with col2:
                risk_level = summary.get('risk_level', 'Medium')
                risk_color = {
                    'High': '🔴',
                    'Medium': '🟡',
                    'Low': '🟢'
                }.get(risk_level, '⚪')
                st.metric("Risk Level", f"{risk_color} {risk_level}")
            
            with col3:
                trend_strength = summary.get('trend_strength', 'Weak')
                st.metric("Trend Strength", trend_strength)
            
            # Key insights
            st.subheader("Key Insights")
            if 'key_insights' in summary:
                for insight in summary['key_insights']:
                    st.info(f"💡 {insight}")
            
            # Recommendations
            st.subheader("Recommendations")
            if 'recommendations' in summary:
                for rec in summary['recommendations']:
                    st.success(f"✅ {rec}")
            
            # Correlation analysis
            if insights and 'correlation_analysis' in insights:
                st.subheader("Feature Correlations")
                corr_matrix = insights['correlation_analysis'].get('correlation_matrix')
                if corr_matrix is not None:
                    corr_chart = visualizer.create_correlation_heatmap(corr_matrix)
                    st.plotly_chart(corr_chart, use_container_width=True,key="corr_chart")
        else:
            st.warning("Market insights not available")
    
    with tab5:
        st.subheader("Summary Dashboard")
        
        # Create comprehensive dashboard
        summary_chart = visualizer.create_summary_dashboard(data, insights)
        st.plotly_chart(summary_chart, use_container_width=True,key="summary_chart")

def display_enhanced_ml_section(data, predictor, ml_task, ml_model, prediction_horizon):
    """Display enhanced machine learning section"""
    
    st.header("🤖 Enhanced Machine Learning Analysis")
    
    # Check if we have data in session state
    if 'processed_data' in st.session_state:
        data = st.session_state['processed_data']
        symbol = st.session_state.get('ml_symbol', 'Unknown')
        st.success(f"✅ Using cached data for {symbol}")
        
        # Add clear cache option
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Data cached: {data.shape[0]} rows, {data.shape[1]} columns")
        with col2:
            if st.button("🗑️ Clear Cache", type="secondary", key="clear_cache_ml"):
                # Clear session state
                for key in ['processed_data', 'ml_insights', 'ml_symbol']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    else:
        st.warning("No processed data available. Please run 'Analyze Market' first.")
        return
    
    # Prepare data for ML
    if 'Price_Change' not in data.columns:
        st.warning("Price change data not available for ML analysis")
        return
    
    # Data preparation - use the preprocessor, not the predictor
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_for_modeling(data, target_column='Price_Change', lookback_periods=5)
    
    if X is None or y is None:
        st.warning("Insufficient data for machine learning analysis")
        return
    
    # Investment Recommendation Section
    st.subheader("💡 Investment Recommendation")
    
    # Calculate basic market sentiment
    recent_close = data['Close'].iloc[-1]
    recent_change = data['Price_Change'].iloc[-1] if 'Price_Change' in data.columns else 0
    ma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
    ma_50 = data['Close'].rolling(window=50).mean().iloc[-1] if len(data) >= 50 else ma_20
    
    # Generate recommendation
    recommendation = generate_investment_recommendation(data, recent_close, recent_change, ma_20, ma_50)
    
    # Display recommendation
    col1, col2, col3 = st.columns(3)
    with col1:
        if recommendation['action'] == 'BUY':
            st.success(f"🟢 **{recommendation['action']}**")
        elif recommendation['action'] == 'SELL':
            st.error(f"🔴 **{recommendation['action']}**")
        else:
            st.warning(f"🟡 **{recommendation['action']}**")
    
    with col2:
        st.metric("Confidence", f"{recommendation['confidence']:.1f}%")
    
    with col3:
        st.metric("Risk Level", recommendation['risk_level'])
    
    # Show reasoning
    st.info(f"**Reasoning:** {recommendation['reasoning']}")
    
    # ML options
    st.subheader("🔧 Machine Learning Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        # Use the passed sidebar values directly (these will update when sidebar changes)
        # Determine task type based on ml_task
        if ml_task in ["Price Prediction", "Volatility Prediction"]:
            task_type = "regression"
        else:  # "Trend Classification", "Risk Assessment"
            task_type = "classification"
        
        # Dynamic model options based on task type
        if task_type == "regression":
            model_options = ["Random Forest", "Linear Regression", "SVR", "Neural Network"]
            model_descriptions = {
                "Random Forest": "🌳 Ensemble method, good for complex patterns",
                "Linear Regression": "📊 Simple linear relationship modeling",
                "SVR": "🎯 Support Vector Regression, good for non-linear data",
                "Neural Network": "🧠 Deep learning, captures complex relationships"
            }
        else:  # classification
            model_options = ["Random Forest", "Logistic Regression", "SVC", "Neural Network"]
            model_descriptions = {
                "Random Forest": "🌳 Ensemble method, good for complex patterns",
                "Logistic Regression": "📊 Logistic regression for binary classification",
                "SVC": "🎯 Support Vector Classification, good for non-linear data",
                "Neural Network": "🧠 Deep learning, captures complex relationships"
            }
        
        # Show current selection from sidebar
        st.info(f"**Selected Task:** {ml_task}")
        st.info(f"**Selected Model:** {ml_model}")
        
        # Show available models for current task
        st.subheader("Available Models for Current Task")
        for model in model_options:
            if model == ml_model:
                st.success(f"✅ {model} - {model_descriptions.get(model, '')}")
            else:
                st.info(f"⚪ {model} - {model_descriptions.get(model, '')}")
        
        # Note about dynamic selection
        st.caption("💡 Change ML Task or Model in the sidebar to see different options")
        
        # Add refresh button to sync with sidebar changes
        if st.button("🔄 Refresh ML Section", key="refresh_ml_section", type="secondary"):
            st.rerun()
    
    with col2:
        # Show task type (read-only, determined by ml_task)
        st.info(f"**Task Type:** {task_type.title()}")
        # Show task description
        task_descriptions = {
            "regression": "📈 Predict continuous price changes",
            "classification": "🔍 Predict up/down price movements"
        }
        st.caption(task_descriptions.get(task_type, ""))
        
        # Show model description
        st.subheader("Model Description")
        st.info(model_descriptions.get(ml_model, "Select a model to see description"))
    
    # Train model button
    if st.button("🚀 Train Model", type="primary", key="train_model_ml"):
        with st.spinner("Training model..."):
            try:
                # Convert display names back to internal format
                model_mapping = {
                    "Random Forest": "random_forest",
                    "Linear Regression": "linear_regression", 
                    "SVR": "svr",
                    "Neural Network": "neural_network",
                    "Logistic Regression": "logistic_regression",
                    "SVC": "svc"
                }
                internal_model_type = model_mapping.get(ml_model, "random_forest")
                
                if task_type == "regression":
                    # For regression, use the converted model_type
                    performance = predictor.train_regression_model(X, y, internal_model_type)
                else:
                    # For classification, use the converted model_type
                    # Convert to classification task
                    y_class = (y > 0).astype(int)
                    performance = predictor.train_classification_model(X, y_class, internal_model_type)
                
                if performance:
                    st.success("Model trained successfully!")
                    
                    # Display performance metrics
                    st.subheader("Model Performance")
                    
                    if task_type == "regression":
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R² Score", f"{performance.get('r2', 0):.4f}")
                        with col2:
                            st.metric("RMSE", f"{performance.get('rmse', 0):.4f}")
                        with col3:
                            st.metric("MAE", f"{performance.get('mae', 0):.4f}")
                        with col4:
                            st.metric("Test Size", performance.get('test_size', 0))
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{performance.get('accuracy', 0):.4f}")
                        with col2:
                            st.metric("Test Size", performance.get('test_size', 0))
                    
                    # Feature importance
                    if predictor.feature_importance:
                        st.subheader("Feature Importance")
                        importance = predictor.get_feature_importance(top_n=10)
                        
                        if importance:
                            importance_df = pd.DataFrame([
                                {"Feature": k, "Importance": v} 
                                for k, v in importance.items()
                            ])
                            
                            fig = px.bar(
                                importance_df, 
                                x="Feature", 
                                y="Importance",
                                title="Top 10 Feature Importance Scores"
                            )
                            st.plotly_chart(fig, use_container_width=True,key="importance_chart")
                    
                    # Make predictions
                    st.subheader("Make Predictions")
                    if st.button("🔮 Predict Next Values", key="predict_values_ml"):
                        # Use last few samples for prediction
                        X_pred = X[-5:]  # Last 5 samples
                        predictions = predictor.make_prediction(X_pred)
                        
                        if predictions is not None:
                            st.success("Predictions generated!")
                            
                            # Display predictions
                            pred_df = pd.DataFrame({
                                "Sample": range(1, len(predictions) + 1),
                                "Predicted Value": predictions
                            })
                            st.dataframe(pred_df)
                            
                            # Plot predictions
                            fig = px.line(
                                pred_df, 
                                x="Sample", 
                                y="Predicted Value",
                                title="Predicted Values",
                                markers=True
                            )
                            st.plotly_chart(fig, use_container_width=True,key="predictions_chart")
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                st.exception(e)
    
    # Summary Section
    st.subheader("📊 Analysis Summary")
    
    # Market overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = data['Close'].iloc[-1]
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        price_change_1d = data['Close'].pct_change().iloc[-1] * 100 if len(data) > 1 else 0
        st.metric("1-Day Change", f"{price_change_1d:.2f}%")
    
    with col3:
        price_change_5d = data['Close'].pct_change(periods=5).iloc[-1] * 100 if len(data) >= 6 else 0
        st.metric("5-Day Change", f"{price_change_5d:.2f}%")
    
    with col4:
        volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100 if len(data) > 1 else 0
        st.metric("Annual Volatility", f"{volatility:.1f}%")
    
    # Key metrics
    st.info("**💡 Key Insights:**")
    st.write(f"• **Trend Analysis**: {'Bullish' if ma_20 > ma_50 else 'Bearish'} trend based on moving averages")
    st.write(f"• **Support Level**: ${data['Low'].rolling(window=20).min().iloc[-1]:.2f} (20-day low)")
    st.write(f"• **Resistance Level**: ${data['High'].rolling(window=20).max().iloc[-1]:.2f} (20-day high)")
    st.write(f"• **Volume Trend**: {'Increasing' if data['Volume'].iloc[-5:].mean() > data['Volume'].iloc[-20:-15].mean() else 'Decreasing'} volume trend")
    
    # Disclaimer
    st.warning("""
    ⚠️ **Disclaimer**: This analysis is for educational and informational purposes only. 
    It should not be considered as financial advice. Always do your own research and 
    consult with a qualified financial advisor before making investment decisions. 
    Past performance does not guarantee future results.
    """)

def display_sidebar_info():
    """Display sidebar information"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 **Application Information**")
    st.sidebar.subheader("About")
    st.sidebar.info("""
    This application provides comprehensive market trend analysis using:
    - Real-time data collection
    - Advanced technical indicators
    - Statistical analysis
    - Machine learning predictions
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Features")
    st.sidebar.markdown("""
    ✅ **Data Collection**: Yahoo Finance integration \n
    ✅ **Technical Analysis**: 20+ indicators \n
    ✅ **Trend Analysis**: Statistical methods \n
    ✅ **Volatility Analysis**: Risk metrics \n
    ✅ **Machine Learning**: Predictive models \n
    ✅ **Interactive Charts**: Plotly visualizations \n
    """)

if __name__ == "__main__":
    try:
        # Display sidebar info
        display_sidebar_info()
        
        # Run main application
        main()
        
    except Exception as e:
        st.error("Application error occurred. Please check the console for details.")
        st.exception(e)
