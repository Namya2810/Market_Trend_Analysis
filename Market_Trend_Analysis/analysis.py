"""
Market Analysis Module for Market Trend Analysis
Handles trend analysis, statistical calculations, and market insights
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """Class for analyzing market trends and patterns"""
    
    def __init__(self):
        self.trend_analysis = {}
        self.volatility_analysis = {}
        self.correlation_analysis = {}
        
    def analyze_trends(self, df, price_column='Close'):
        """
        Analyze price trends using various methods
        
        Args:
            df (pd.DataFrame): Input dataframe with price data
            price_column (str): Column name for price data
            
        Returns:
            dict: Dictionary containing trend analysis results
        """
        try:
            logger.info("Analyzing price trends")
            
            if price_column not in df.columns:
                logger.error(f"Price column {price_column} not found")
                return {}
            
            # Calculate basic trend metrics
            prices = df[price_column].dropna()
            
            # Linear regression trend
            x = np.arange(len(prices))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
            
            # Trend strength
            trend_strength = abs(r_value)
            trend_direction = "Upward" if slope > 0 else "Downward"
            
            # Moving average trends
            ma_20 = prices.rolling(window=20).mean()
            ma_50 = prices.rolling(window=50).mean()
            
            # Current trend based on moving averages
            current_ma_trend = "Bullish" if ma_20.iloc[-1] > ma_50.iloc[-1] else "Bearish"
            
            # Price momentum
            momentum_5 = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100 if len(prices) >= 5 else 0
            momentum_20 = (prices.iloc[-1] / prices.iloc[-20] - 1) * 100 if len(prices) >= 20 else 0
            
            # Support and resistance levels
            support_level = prices.rolling(window=20).min().iloc[-1]
            resistance_level = prices.rolling(window=20).max().iloc[-1]
            
            # Trend analysis results
            trend_analysis = {
                'linear_trend_slope': slope,
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'current_ma_trend': current_ma_trend,
                'momentum_5d': momentum_5,
                'momentum_20d': momentum_20,
                'support_level': support_level,
                'resistance_level': resistance_level,
                'price_change_ytd': (prices.iloc[-1] / prices.iloc[0] - 1) * 100 if len(prices) > 0 else 0,
                'trend_consistency': self._calculate_trend_consistency(prices)
            }
            
            self.trend_analysis = trend_analysis
            logger.info("Trend analysis completed successfully")
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            return {}
    
    def _calculate_trend_consistency(self, prices, window=20):
        """Calculate trend consistency over a rolling window"""
        try:
            if len(prices) < window:
                return 0
            
            consistency_scores = []
            for i in range(window, len(prices)):
                window_prices = prices.iloc[i-window:i]
                x = np.arange(len(window_prices))
                slope, _, r_value, _, _ = stats.linregress(x, window_prices)
                
                # Score based on R-squared and slope consistency
                score = (r_value ** 2) * (1 if slope > 0 else -1)
                consistency_scores.append(score)
            
            return np.mean(consistency_scores) if consistency_scores else 0
            
        except Exception as e:
            logger.error(f"Error calculating trend consistency: {str(e)}")
            return 0
    
    def analyze_volatility(self, df, price_column='Close', volume_column='Volume'):
        """
        Analyze price and volume volatility
        
        Args:
            df (pd.DataFrame): Input dataframe
            price_column (str): Column name for price data
            volume_column (str): Column name for volume data
            
        Returns:
            dict: Dictionary containing volatility analysis results
        """
        try:
            logger.info("Analyzing volatility patterns")
            
            if price_column not in df.columns:
                logger.error(f"Price column {price_column} not found")
                return {}
            
            prices = df[price_column].dropna()
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Volatility metrics
            volatility_metrics = {
                'daily_volatility': returns.std() * np.sqrt(252),  # Annualized
                'annualized_volatility': returns.std() * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(prices),
                'var_95': np.percentile(returns, 5),  # Value at Risk 95%
                'var_99': np.percentile(returns, 1),  # Value at Risk 99%
                'volatility_of_volatility': returns.rolling(window=20).std().std(),
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns),
                'volatility_regime': self._identify_volatility_regime(returns)
            }
            
            # Volume volatility analysis
            if volume_column in df.columns:
                volume = df[volume_column].dropna()
                volume_returns = volume.pct_change().dropna()
                
                volume_metrics = {
                    'volume_volatility': volume_returns.std(),
                    'volume_trend': self._calculate_volume_trend(volume),
                    'price_volume_correlation': returns.corr(volume_returns)
                }
                volatility_metrics.update(volume_metrics)
            
            self.volatility_analysis = volatility_metrics
            logger.info("Volatility analysis completed successfully")
            return volatility_metrics
            
        except Exception as e:
            logger.error(f"Error in volatility analysis: {str(e)}")
            return {}
    
    def _calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown from peak"""
        try:
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            return drawdown.min()
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0
    
    def _identify_volatility_regime(self, returns, window=20):
        """Identify current volatility regime"""
        try:
            if len(returns) < window:
                return "Unknown"
            
            current_vol = returns.rolling(window=window).std().iloc[-1]
            historical_vol = returns.rolling(window=window).std().mean()
            
            if current_vol > historical_vol * 1.5:
                return "High Volatility"
            elif current_vol < historical_vol * 0.5:
                return "Low Volatility"
            else:
                return "Normal Volatility"
                
        except Exception as e:
            logger.error(f"Error identifying volatility regime: {str(e)}")
            return "Unknown"
    
    def _calculate_volume_trend(self, volume):
        """Calculate volume trend direction"""
        try:
            if len(volume) < 20:
                return "Insufficient Data"
            
            recent_volume = volume.iloc[-20:].mean()
            historical_volume = volume.iloc[:-20].mean() if len(volume) > 20 else recent_volume
            
            if recent_volume > historical_volume * 1.2:
                return "Increasing"
            elif recent_volume < historical_volume * 0.8:
                return "Decreasing"
            else:
                return "Stable"
                
        except Exception as e:
            logger.error(f"Error calculating volume trend: {str(e)}")
            return "Unknown"
    
    def analyze_correlations(self, df, exclude_columns=None):
        """
        Analyze correlations between different features
        
        Args:
            df (pd.DataFrame): Input dataframe
            exclude_columns (list): Columns to exclude from correlation analysis
            
        Returns:
            dict: Dictionary containing correlation analysis results
        """
        try:
            logger.info("Analyzing feature correlations")
            
            if exclude_columns is None:
                exclude_columns = ['Date', 'Symbol']
            
            # Select numeric columns for correlation analysis
            numeric_df = df.select_dtypes(include=[np.number])
            numeric_df = numeric_df.drop(columns=[col for col in exclude_columns if col in numeric_df.columns])
            
            if numeric_df.empty:
                logger.warning("No numeric columns available for correlation analysis")
                return {}
            
            # Calculate correlation matrix
            correlation_matrix = numeric_df.corr()
            
            # Find highly correlated features
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        high_correlations.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            # Principal Component Analysis
            pca = PCA()
            pca.fit(numeric_df.fillna(0))
            
            # Calculate explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Find number of components for 95% variance
            n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
            
            correlation_analysis = {
                'correlation_matrix': correlation_matrix,
                'high_correlations': high_correlations,
                'pca_explained_variance': explained_variance,
                'cumulative_variance': cumulative_variance,
                'n_components_95': n_components_95,
                'total_features': len(numeric_df.columns),
                'redundant_features': len(high_correlations)
            }
            
            self.correlation_analysis = correlation_analysis
            logger.info("Correlation analysis completed successfully")
            return correlation_analysis
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return {}
    
    def perform_market_regime_analysis(self, df, price_column='Close'):
        """
        Perform market regime analysis using clustering
        
        Args:
            df (pd.DataFrame): Input dataframe
            price_column (str): Column name for price data
            
        Returns:
            dict: Dictionary containing regime analysis results
        """
        try:
            logger.info("Performing market regime analysis")
            
            if price_column not in df.columns:
                logger.error(f"Price column {price_column} not found")
                return {}
            
            prices = df[price_column].dropna()
            
            # Calculate features for regime analysis
            returns = prices.pct_change().dropna()
            volatility = returns.rolling(window=20).std().dropna()
            momentum = prices.rolling(window=20).mean().pct_change().dropna()
            
            # Align all series
            min_length = min(len(returns), len(volatility), len(momentum))
            returns = returns.iloc[-min_length:]
            volatility = volatility.iloc[-min_length:]
            momentum = momentum.iloc[-min_length:]
            
            # Create feature matrix
            features = np.column_stack([
                returns.values,
                volatility.values,
                momentum.values
            ])
            
            # Remove rows with NaN values
            features = features[~np.isnan(features).any(axis=1)]
            
            if len(features) < 10:
                logger.warning("Insufficient data for regime analysis")
                return {}
            
            # Perform clustering
            n_clusters = 3  # Bull, Bear, Sideways
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            
            # Analyze each regime
            regimes = {}
            for i in range(n_clusters):
                regime_mask = cluster_labels == i
                regime_features = features[regime_mask]
                
                regimes[f'Regime_{i+1}'] = {
                    'size': int(regime_mask.sum()),
                    'avg_return': float(np.mean(regime_features[:, 0])),
                    'avg_volatility': float(np.mean(regime_features[:, 1])),
                    'avg_momentum': float(np.mean(regime_features[:, 2])),
                    'duration_estimate': int(regime_mask.sum() / len(features) * 252)  # Annualized
                }
            
            # Identify current regime
            current_features = features[-1:] if len(features) > 0 else None
            current_regime = None
            if current_features is not None:
                current_cluster = kmeans.predict(current_features)[0]
                current_regime = f'Regime_{current_cluster + 1}'
            
            regime_analysis = {
                'regimes': regimes,
                'current_regime': current_regime,
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'feature_names': ['Returns', 'Volatility', 'Momentum']
            }
            
            logger.info("Market regime analysis completed successfully")
            return regime_analysis
            
        except Exception as e:
            logger.error(f"Error in market regime analysis: {str(e)}")
            return {}
    
    def generate_market_insights(self, df, price_column='Close'):
        """
        Generate comprehensive market insights
        
        Args:
            df (pd.DataFrame): Input dataframe
            price_column (str): Column name for price data
            
        Returns:
            dict: Dictionary containing market insights
        """
        try:
            logger.info("Generating market insights")
            
            insights = {}
            
            # Perform all analyses
            trend_insights = self.analyze_trends(df, price_column)
            volatility_insights = self.analyze_volatility(df, price_column)
            correlation_insights = self.analyze_correlations(df)
            regime_insights = self.perform_market_regime_analysis(df, price_column)
            
            # Combine insights
            insights.update({
                'trend_analysis': trend_insights,
                'volatility_analysis': volatility_insights,
                'correlation_analysis': correlation_insights,
                'regime_analysis': regime_insights
            })
            
            # Generate summary insights
            summary_insights = self._generate_summary_insights(insights)
            insights['summary'] = summary_insights
            
            logger.info("Market insights generated successfully")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating market insights: {str(e)}")
            return {}
    
    def _generate_summary_insights(self, insights):
        """Generate summary insights from all analyses"""
        try:
            summary = {
                'overall_market_sentiment': 'Neutral',
                'risk_level': 'Medium',
                'trend_strength': 'Weak',
                'key_insights': [],
                'recommendations': []
            }
            
            # Determine overall sentiment
            if 'trend_analysis' in insights and insights['trend_analysis']:
                trend = insights['trend_analysis']
                if trend.get('trend_direction') == 'Upward' and trend.get('trend_strength', 0) > 0.7:
                    summary['overall_market_sentiment'] = 'Bullish'
                elif trend.get('trend_direction') == 'Downward' and trend.get('trend_strength', 0) > 0.7:
                    summary['overall_market_sentiment'] = 'Bearish'
            
            # Determine risk level
            if 'volatility_analysis' in insights and insights['volatility_analysis']:
                vol = insights['volatility_analysis']
                if vol.get('daily_volatility', 0) > 0.3:
                    summary['risk_level'] = 'High'
                elif vol.get('daily_volatility', 0) < 0.15:
                    summary['risk_level'] = 'Low'
            
            # Determine trend strength
            if 'trend_analysis' in insights and insights['trend_analysis']:
                trend_strength = insights['trend_analysis'].get('trend_strength', 0)
                if trend_strength > 0.8:
                    summary['trend_strength'] = 'Strong'
                elif trend_strength > 0.5:
                    summary['trend_strength'] = 'Moderate'
            
            # Generate key insights
            if 'trend_analysis' in insights and insights['trend_analysis']:
                trend = insights['trend_analysis']
                summary['key_insights'].append(f"Market showing {trend.get('trend_direction', 'Unknown')} trend")
                summary['key_insights'].append(f"Trend strength: {trend.get('trend_strength', 0):.2f}")
            
            if 'volatility_analysis' in insights and insights['volatility_analysis']:
                vol = insights['volatility_analysis']
                summary['key_insights'].append(f"Current volatility regime: {vol.get('volatility_regime', 'Unknown')}")
            
            # Generate recommendations
            if summary['overall_market_sentiment'] == 'Bullish':
                summary['recommendations'].append("Consider long positions with proper risk management")
            elif summary['overall_market_sentiment'] == 'Bearish':
                summary['recommendations'].append("Consider defensive positions or short opportunities")
            
            if summary['risk_level'] == 'High':
                summary['recommendations'].append("Implement strict risk management and position sizing")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary insights: {str(e)}")
            return {}

def main():
    """Test function for the analysis module"""
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
    
    analyzer = MarketAnalyzer()
    
    # Test trend analysis
    print("Testing trend analysis...")
    trend_results = analyzer.analyze_trends(sample_data)
    print(f"Trend analysis completed: {len(trend_results)} metrics")
    
    # Test volatility analysis
    print("\nTesting volatility analysis...")
    volatility_results = analyzer.analyze_volatility(sample_data)
    print(f"Volatility analysis completed: {len(volatility_results)} metrics")
    
    # Test correlation analysis
    print("\nTesting correlation analysis...")
    correlation_results = analyzer.analyze_correlations(sample_data)
    print(f"Correlation analysis completed: {len(correlation_results)} metrics")
    
    # Test regime analysis
    print("\nTesting regime analysis...")
    regime_results = analyzer.perform_market_regime_analysis(sample_data)
    print(f"Regime analysis completed: {len(regime_results)} metrics")
    
    # Test comprehensive insights
    print("\nTesting comprehensive insights...")
    insights = analyzer.generate_market_insights(sample_data)
    print(f"Comprehensive insights generated: {len(insights)} sections")

if __name__ == "__main__":
    main()


