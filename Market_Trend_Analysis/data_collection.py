"""
Data Collection Module for Market Trend Analysis
Handles fetching market data from various sources
"""

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataCollector:
    """Class for collecting market data from various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_stock_data(self, symbol, period="1y", interval="1d"):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
            period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            pd.DataFrame: Stock data with OHLCV information
        """
        try:
            logger.info(f"Fetching data for {symbol} with period {period} and interval {interval}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Reset index to make Date a column
            data = data.reset_index()
            data['Symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_stocks(self, symbols, period="1y", interval="1d"):
        """
        Fetch data for multiple stocks
        
        Args:
            symbols (list): List of stock symbols
            period (str): Time period
            interval (str): Data interval
        
        Returns:
            dict: Dictionary with symbol as key and data as value
        """
        data_dict = {}
        
        for symbol in symbols:
            data = self.get_stock_data(symbol, period, interval)
            if not data.empty:
                data_dict[symbol] = data
            time.sleep(0.1)  # Rate limiting
        
        return data_dict
    
    def get_market_indices(self):
        """
        Fetch major market indices data
        
        Returns:
            dict: Dictionary with index data
        """
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^VIX': 'VIX Volatility Index'
        }
        
        indices_data = {}
        for symbol, name in indices.items():
            data = self.get_stock_data(symbol, period="1y", interval="1d")
            if not data.empty:
                indices_data[name] = data
        
        return indices_data
    
    def get_crypto_data(self, symbol, period="1y", interval="1d"):
        """
        Fetch cryptocurrency data
        
        Args:
            symbol (str): Crypto symbol (e.g., 'BTC-USD', 'ETH-USD')
            period (str): Time period
            interval (str): Data interval
        
        Returns:
            pd.DataFrame: Crypto data
        """
        return self.get_stock_data(symbol, period, interval)
    
    def get_forex_data(self, symbol, period="1y", interval="1d"):
        """
        Fetch forex data
        
        Args:
            symbol (str): Forex pair (e.g., 'EURUSD=X', 'GBPUSD=X')
            period (str): Time period
            interval (str): Data interval
        
        Returns:
            pd.DataFrame: Forex data
        """
        return self.get_stock_data(symbol, period, interval)
    
    def scrape_market_news(self, source="yahoo", limit=10):
        """
        Scrape market news from various sources
        
        Args:
            source (str): News source ('yahoo', 'marketwatch')
            limit (int): Number of news articles to fetch
        
        Returns:
            list: List of news articles
        """
        news = []
        
        try:
            if source == "yahoo":
                url = "https://finance.yahoo.com/news/"
                response = self.session.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # This is a simplified example - actual implementation would need
                # to handle dynamic content and site structure changes
                articles = soup.find_all('h3', class_='Mb(5px)')[:limit]
                
                for article in articles:
                    title = article.get_text(strip=True)
                    if title:
                        news.append({
                            'title': title,
                            'source': 'Yahoo Finance',
                            'timestamp': datetime.now().isoformat()
                        })
            
            logger.info(f"Scraped {len(news)} news articles from {source}")
            
        except Exception as e:
            logger.error(f"Error scraping news from {source}: {str(e)}")
        
        return news
    
    def get_economic_calendar(self):
        """
        Get economic calendar events (placeholder for future implementation)
        
        Returns:
            list: Economic events
        """
        # This would typically integrate with APIs like Trading Economics
        # For now, return sample data
        return [
            {
                'date': '2024-01-15',
                'event': 'Federal Reserve Meeting',
                'impact': 'High',
                'currency': 'USD'
            },
            {
                'date': '2024-01-20',
                'event': 'Non-Farm Payrolls',
                'impact': 'High',
                'currency': 'USD'
            }
        ]

def main():
    """Test function for the data collection module"""
    collector = MarketDataCollector()
    
    # Test stock data collection
    print("Testing stock data collection...")
    aapl_data = collector.get_stock_data('AAPL', period='1mo', interval='1d')
    print(f"AAPL data shape: {aapl_data.shape}")
    
    # Test multiple stocks
    print("\nTesting multiple stocks...")
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    multi_data = collector.get_multiple_stocks(symbols, period='1mo')
    for symbol, data in multi_data.items():
        print(f"{symbol}: {len(data)} records")
    
    # Test market indices
    print("\nTesting market indices...")
    indices_data = collector.get_market_indices()
    for name, data in indices_data.items():
        print(f"{name}: {len(data)} records")

if __name__ == "__main__":
    main()
