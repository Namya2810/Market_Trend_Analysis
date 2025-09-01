"""
Data Preprocessing Module for Market Trend Analysis
Handles data cleaning, feature engineering, and data preparation
"""

import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class for preprocessing market data"""
    
    def __init__(self):
        self.scaler = None
        self.imputer = None
        
    def clean_data(self, df):
        """
        Clean the input dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        try:
            logger.info("Starting data cleaning process")
            
            # Make a copy to avoid modifying original data
            df_clean = df.copy()
            
            # Remove duplicates
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            if len(df_clean) < initial_rows:
                logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
            
            # Handle missing values
            df_clean = self._handle_missing_values(df_clean)
            
            # Remove rows with extreme outliers
            df_clean = self._remove_outliers(df_clean)
            
            # Sort by date if available
            if 'Date' in df_clean.columns:
                df_clean = df_clean.sort_values('Date').reset_index(drop=True)
            
            logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
            return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataframe"""
        try:
            # For OHLCV data, forward fill is often appropriate
            if 'Open' in df.columns and 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
                # Forward fill for OHLC data
                df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].fillna(method='ffill')
                
                # For Volume, fill with 0 or median
                if 'Volume' in df.columns:
                    df['Volume'] = df['Volume'].fillna(0)
            
            # For other numeric columns, use forward fill then backward fill
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill')
            
            # For remaining missing values, use simple imputation
            if df.isnull().sum().sum() > 0:
                self.imputer = SimpleImputer(strategy='mean')
                df_imputed = pd.DataFrame(
                    self.imputer.fit_transform(df.select_dtypes(include=[np.number])),
                    columns=df.select_dtypes(include=[np.number]).columns,
                    index=df.index
                )
                
                # Update numeric columns
                for col in df_imputed.columns:
                    df[col] = df_imputed[col]
            
            logger.info("Missing values handled successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return df
    
    def _remove_outliers(self, df, threshold=3):
        """Remove extreme outliers using z-score method"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in ['Date', 'Symbol']:  # Skip non-numeric columns
                    continue
                    
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > threshold
                
                if outliers.sum() > 0:
                    logger.info(f"Removing {outliers.sum()} outliers from column {col}")
                    df = df[~outliers]
            
            return df
            
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            return df
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe with OHLCV data
            
        Returns:
            pd.DataFrame: Dataframe with technical indicators
        """
        try:
            logger.info("Adding technical indicators")
            df_indicators = df.copy()
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df_indicators.columns for col in required_columns):
                logger.warning("Missing required OHLCV columns for technical indicators")
                logger.warning(f"Available columns: {list(df_indicators.columns)}")
                return df_indicators
            
            # Trend Indicators
            df_indicators['SMA_20'] = ta.trend.sma_indicator(df_indicators['Close'], window=20)
            df_indicators['SMA_50'] = ta.trend.sma_indicator(df_indicators['Close'], window=50)
            df_indicators['EMA_12'] = ta.trend.ema_indicator(df_indicators['Close'], window=12)
            df_indicators['EMA_26'] = ta.trend.ema_indicator(df_indicators['Close'], window=26)
            
            # MACD
            df_indicators['MACD'] = ta.trend.macd_diff(df_indicators['Close'])
            df_indicators['MACD_Signal'] = ta.trend.macd_signal(df_indicators['Close'])
            
            # RSI
            df_indicators['RSI'] = ta.momentum.rsi(df_indicators['Close'], window=14)
            
            # Bollinger Bands
            bb_high = ta.volatility.bollinger_hband(df_indicators['Close'])
            bb_low = ta.volatility.bollinger_lband(df_indicators['Close'])
            bb_mid = ta.volatility.bollinger_mavg(df_indicators['Close'])
            
            df_indicators['BB_High'] = bb_high
            df_indicators['BB_Low'] = bb_low
            df_indicators['BB_Mid'] = bb_mid
            df_indicators['BB_Width'] = bb_high - bb_low
            
            # Volume Indicators
            df_indicators['Volume_SMA'] = ta.volume.volume_sma(df_indicators['Close'], df_indicators['Volume'])
            df_indicators['OBV'] = ta.volume.on_balance_volume(df_indicators['Close'], df_indicators['Volume'])
            
            # Volatility Indicators
            df_indicators['ATR'] = ta.volatility.average_true_range(df_indicators['High'], df_indicators['Low'], df_indicators['Close'])
            
            # Price-based features
            logger.info("Calculating Price_Change...")
            df_indicators['Price_Change'] = df_indicators['Close'].pct_change()
            logger.info(f"Price_Change created. Shape: {df_indicators['Price_Change'].shape}")
            logger.info(f"Price_Change sample values: {df_indicators['Price_Change'].head().tolist()}")
            
            df_indicators['Price_Change_5'] = df_indicators['Close'].pct_change(periods=5)
            df_indicators['Price_Change_20'] = df_indicators['Close'].pct_change(periods=20)
            
            # High-Low spread
            df_indicators['HL_Spread'] = (df_indicators['High'] - df_indicators['Low']) / df_indicators['Close']
            
            # Gap analysis
            df_indicators['Gap'] = df_indicators['Open'] - df_indicators['Close'].shift(1)
            df_indicators['Gap_Pct'] = df_indicators['Gap'] / df_indicators['Close'].shift(1)
            
            logger.info(f"Final columns after adding indicators: {list(df_indicators.columns)}")
            logger.info(f"Added {len(df_indicators.columns) - len(df.columns)} technical indicators")
            return df_indicators
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            logger.error(f"Error details: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Even if other indicators fail, ensure Price_Change is created
            logger.info("Attempting to create essential Price_Change columns...")
            try:
                df_indicators['Price_Change'] = df_indicators['Close'].pct_change()
                df_indicators['Price_Change_5'] = df_indicators['Close'].pct_change(periods=5)
                df_indicators['Price_Change_20'] = df_indicators['Close'].pct_change(periods=20)
                logger.info("Essential Price_Change columns created successfully")
            except Exception as price_error:
                logger.error(f"Failed to create Price_Change: {str(price_error)}")
            
            # Return the enhanced data even if some indicators failed
            return df_indicators
    
    def create_features(self, df):
        """
        Create additional features for analysis
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with additional features
        """
        try:
            logger.info("Creating additional features")
            df_features = df.copy()
            
            # Time-based features
            if 'Date' in df_features.columns:
                df_features['Date'] = pd.to_datetime(df_features['Date'])
                df_features['Year'] = df_features['Date'].dt.year
                df_features['Month'] = df_features['Date'].dt.month
                df_features['Day'] = df_features['Date'].dt.day
                df_features['DayOfWeek'] = df_features['Date'].dt.dayofweek
                df_features['Quarter'] = df_features['Date'].dt.quarter
                
                # Market session (assuming US market hours)
                df_features['Hour'] = df_features['Date'].dt.hour
                df_features['Is_Market_Hours'] = ((df_features['Hour'] >= 9) & (df_features['Hour'] < 16)).astype(int)
            
            # Rolling statistics
            if 'Close' in df_features.columns:
                for window in [5, 10, 20, 50]:
                    df_features[f'Close_Rolling_Mean_{window}'] = df_features['Close'].rolling(window=window).mean()
                    df_features[f'Close_Rolling_Std_{window}'] = df_features['Close'].rolling(window=window).std()
                    df_features[f'Close_Rolling_Min_{window}'] = df_features['Close'].rolling(window=window).min()
                    df_features[f'Close_Rolling_Max_{window}'] = df_features['Close'].rolling(window=window).max()
            
            # Volatility features
            if 'Price_Change' in df_features.columns:
                for window in [5, 10, 20]:
                    df_features[f'Volatility_{window}'] = df_features['Price_Change'].rolling(window=window).std()
            
            # Momentum features
            if 'Close' in df_features.columns:
                for period in [5, 10, 20]:
                    df_features[f'Momentum_{period}'] = df_features['Close'] / df_features['Close'].shift(period) - 1
            
            logger.info(f"Created additional features. Final shape: {df_features.shape}")
            return df_features
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return df
    
    def normalize_data(self, df, method='standard', exclude_columns=None):
        """
        Normalize numerical data
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Normalization method ('standard', 'minmax')
            exclude_columns (list): Columns to exclude from normalization
            
        Returns:
            pd.DataFrame: Normalized dataframe
        """
        try:
            logger.info(f"Normalizing data using {method} method")
            
            if exclude_columns is None:
                exclude_columns = ['Date', 'Symbol']
            
            # Select only numeric columns for normalization
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            columns_to_normalize = [col for col in numeric_columns if col not in exclude_columns]
            
            if not columns_to_normalize:
                logger.warning("No columns to normalize")
                return df
            
            df_normalized = df.copy()
            
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError("Method must be 'standard' or 'minmax'")
            
            # Fit and transform the data
            normalized_data = self.scaler.fit_transform(df_normalized[columns_to_normalize])
            
            # Update the dataframe
            df_normalized[columns_to_normalize] = normalized_data
            
            logger.info(f"Data normalized successfully for {len(columns_to_normalize)} columns")
            return df_normalized
            
        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            return df
    
    def prepare_for_modeling(self, df, target_column='Price_Change', lookback_periods=5):
        """
        Prepare data for machine learning models
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Target variable column
            lookback_periods (int): Number of previous periods to use as features
            
        Returns:
            tuple: (X, y) features and target arrays
        """
        try:
            logger.info("Preparing data for modeling")
            
            # Check if target column exists
            if target_column not in df.columns:
                logger.error(f"Target column {target_column} not found")
                logger.error(f"Available columns: {list(df.columns)}")
                return None, None
            
            # Remove rows with NaN values in the target column only
            df_clean = df.dropna(subset=[target_column])
            
            if len(df_clean) == 0:
                logger.error("No data remaining after removing NaN values from target column")
                return None, None
            
            logger.info(f"Data shape after cleaning: {df_clean.shape}")
            
            # Create lagged features
            feature_columns = [col for col in df_clean.columns if col not in ['Date', 'Symbol', target_column]]
            
            X_data = []
            y_data = []
            
            for i in range(lookback_periods, len(df_clean)):
                # Features from previous periods
                features = []
                for j in range(lookback_periods):
                    features.extend(df_clean[feature_columns].iloc[i-j-1].values)
                
                X_data.append(features)
                y_data.append(df_clean[target_column].iloc[i])
            
            X = np.array(X_data)
            y = np.array(y_data)
            
            logger.info(f"Final X shape: {X.shape}, y shape: {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data for modeling: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None

def main():
    """Test function for the preprocessing module"""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100),
        'Symbol': ['AAPL'] * 100
    })
    
    preprocessor = DataPreprocessor()
    
    # Test data cleaning
    print("Testing data cleaning...")
    cleaned_data = preprocessor.clean_data(sample_data)
    print(f"Cleaned data shape: {cleaned_data.shape}")
    
    # Test technical indicators
    print("\nTesting technical indicators...")
    data_with_indicators = preprocessor.add_technical_indicators(cleaned_data)
    print(f"Data with indicators shape: {data_with_indicators.shape}")
    
    # Test feature creation
    print("\nTesting feature creation...")
    data_with_features = preprocessor.create_features(data_with_indicators)
    print(f"Data with features shape: {data_with_features.shape}")
    
    # Test normalization
    print("\nTesting normalization...")
    normalized_data = preprocessor.normalize_data(data_with_features)
    print(f"Normalized data shape: {normalized_data.shape}")
    
    # Test modeling preparation
    print("\nTesting modeling preparation...")
    X, y = preprocessor.prepare_for_modeling(normalized_data)
    if X is not None:
        print(f"X shape: {X.shape}, y shape: {y.shape}")

if __name__ == "__main__":
    main()


