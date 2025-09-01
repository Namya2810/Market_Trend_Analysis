# 📈 Market Trend Analysis

A comprehensive Streamlit-based application for analyzing market trends, performing technical analysis, and making predictions using machine learning models.

## 🚀 Features

### Core Functionality
- **Real-time Data Collection**: Fetch market data from Yahoo Finance API
- **Technical Analysis**: 20+ technical indicators including RSI, MACD, Bollinger Bands
- **Trend Analysis**: Statistical trend detection and strength measurement
- **Volatility Analysis**: Risk metrics and volatility regime identification
- **Machine Learning**: Predictive models for price and trend forecasting
- **Interactive Visualizations**: Beautiful charts using Plotly

### Technical Indicators
- Moving Averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume indicators
- ATR (Average True Range)
- Support and Resistance levels

### Machine Learning Models
- **Regression Models**: Random Forest, Linear Regression, SVR, Neural Networks
- **Classification Models**: Trend direction prediction
- **Feature Engineering**: Advanced feature creation and selection
- **Hyperparameter Tuning**: Automated model optimization
- **Model Persistence**: Save and load trained models

## 🏗️ Project Structure

```
Market_Trend_Analysis/
├── app.py                 # Main Streamlit application
├── data_collection.py     # Data fetching from various sources
├── preprocessing.py       # Data cleaning and feature engineering
├── analysis.py           # Market trend and statistical analysis
├── visualization.py      # Chart and plot creation
├── model.py             # Machine learning models and predictions
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Market_Trend_Analysis
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🎯 Usage

### Getting Started

1. **Launch the Application**
   - Run `streamlit run app.py`
   - Open your browser to the displayed URL

2. **Configure Analysis**
   - Select data source (Yahoo Finance or Sample Data)
   - Enter stock symbol (e.g., AAPL, MSFT, GOOGL)
   - Choose time period and interval
   - Enable/disable technical indicators and ML features

3. **Run Analysis**
   - Click "🚀 Analyze Market" button
   - Wait for data processing and analysis
   - Explore results in different tabs

### Available Views

#### 📊 Price Chart
- Interactive candlestick charts
- Technical indicators overlay
- Volume analysis
- Customizable display options

#### 📈 Trend Analysis
- Trend direction and strength
- Moving average analysis
- Momentum indicators
- Support/resistance levels

#### 📉 Volatility Analysis
- Daily and annualized volatility
- Maximum drawdown
- Value at Risk (VaR)
- Volatility regime identification

#### 🔍 Market Insights
- Market sentiment analysis
- Risk assessment
- Key insights and recommendations
- Feature correlation analysis

#### 📋 Summary Dashboard
- Comprehensive market overview
- Performance metrics
- Risk indicators
- Technical summary

#### 🤖 Machine Learning
- Model training and evaluation
- Performance metrics
- Feature importance analysis
- Prediction generation

## 🔧 Configuration

### Data Sources
- **Yahoo Finance**: Real-time market data
- **Sample Data**: Generated data for testing and demonstration

### Time Periods
- 1 month (1mo)
- 3 months (3mo)
- 6 months (6mo)
- 1 year (1y)
- 2 years (2y)
- 5 years (5y)

### Intervals
- Daily (1d)
- Weekly (1wk)
- Monthly (1mo)

### Analysis Options
- Technical indicators display
- Volume analysis
- Machine learning features
- Custom chart options

## 🧠 Machine Learning Features

### Model Types
- **Random Forest**: Robust ensemble method
- **Linear Regression**: Simple linear relationships
- **Support Vector Regression**: Non-linear patterns
- **Neural Networks**: Complex pattern recognition

### Training Process
1. **Data Preparation**: Feature engineering and scaling
2. **Model Selection**: Choose algorithm and parameters
3. **Training**: Fit model to historical data
4. **Evaluation**: Performance metrics and validation
5. **Prediction**: Generate future forecasts

### Performance Metrics
- **Regression**: R², RMSE, MAE
- **Classification**: Accuracy, Precision, Recall
- **Feature Importance**: Identify key predictors

## 📊 Technical Indicators

### Trend Indicators
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- MACD and Signal Line
- Trend Strength Index

### Momentum Indicators
- Relative Strength Index (RSI)
- Price Momentum
- Volume Momentum
- Rate of Change

### Volatility Indicators
- Bollinger Bands
- Average True Range (ATR)
- Volatility Ratio
- Price Channels

### Volume Indicators
- Volume SMA
- On-Balance Volume (OBV)
- Volume Price Trend
- Money Flow Index

## 🚨 Risk Disclaimer

**⚠️ Important**: This application is for educational and research purposes only. It is not intended to provide financial advice or recommendations. 

- Past performance does not guarantee future results
- Market predictions are inherently uncertain
- Always conduct your own research and analysis
- Consult with financial professionals before making investment decisions
- Use at your own risk

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest

# Check code quality
flake8 .
black .
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Yahoo Finance**: For providing market data APIs
- **Streamlit**: For the amazing web application framework
- **Plotly**: For interactive charting capabilities
- **Scikit-learn**: For machine learning algorithms
- **TA-Lib**: For technical analysis indicators

## 📞 Support

If you encounter any issues or have questions:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information
4. Include error messages and system details

## 🔄 Updates

### Version 1.0.0
- Initial release with core functionality
- Basic technical analysis
- Machine learning models
- Interactive visualizations

### Planned Features
- Additional data sources
- More technical indicators
- Advanced ML models
- Portfolio analysis
- Backtesting capabilities
- Real-time alerts

---

**Happy Trading! 📈💰**

*Remember: The best investment is in knowledge and understanding.*


