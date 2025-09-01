"""
Data Visualization Module for Market Trend Analysis
Handles creation of interactive charts and plots
"""

import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketVisualizer:
    """Class for creating market data visualizations"""
    
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
    def create_price_chart(self, df, symbol=None, show_volume=True, show_indicators=True):
        """
        Create an interactive price chart with candlesticks
        
        Args:
            df (pd.DataFrame): Dataframe with OHLCV data
            symbol (str): Symbol to display in title
            show_volume (bool): Whether to show volume bars
            show_indicators (bool): Whether to show technical indicators
            
        Returns:
            plotly.graph_objects.Figure: Interactive price chart
        """
        try:
            logger.info("Creating price chart")
            
            if df.empty:
                logger.warning("Empty dataframe provided for price chart")
                return go.Figure()
            
            # Determine subplot layout
            if show_volume and show_indicators:
                fig = sp.make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('Price Chart', 'Technical Indicators', 'Volume'),
                    row_heights=[0.6, 0.2, 0.2]
                )
            elif show_volume:
                fig = sp.make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('Price Chart', 'Volume'),
                    row_heights=[0.7, 0.3]
                )
            else:
                fig = sp.make_subplots(
                    rows=1, cols=1,
                    subplot_titles=('Price Chart',)
                )
            
            # Add candlestick chart
            candlestick = go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC',
                increasing_line_color=self.colors['success'],
                decreasing_line_color=self.colors['danger']
            )
            
            fig.add_trace(candlestick, row=1, col=1)
            
            # Add moving averages if available
            if show_indicators:
                if 'SMA_20' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=df['SMA_20'],
                            mode='lines',
                            name='SMA 20',
                            line=dict(color=self.colors['primary'], width=1)
                        ),
                        row=1, col=1
                    )
                
                if 'SMA_50' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=df['SMA_50'],
                            mode='lines',
                            name='SMA 50',
                            line=dict(color=self.colors['secondary'], width=1)
                        ),
                        row=1, col=1
                    )
                
                # Add Bollinger Bands if available
                if all(col in df.columns for col in ['BB_High', 'BB_Low', 'BB_Mid']):
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=df['BB_High'],
                            mode='lines',
                            name='BB Upper',
                            line=dict(color=self.colors['info'], width=1, dash='dash'),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=df['BB_Low'],
                            mode='lines',
                            name='BB Lower',
                            line=dict(color=self.colors['info'], width=1, dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(23, 162, 184, 0.1)',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
            
            # Add technical indicators subplot
            if show_indicators and show_volume:
                # RSI
                if 'RSI' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=df['RSI'],
                            mode='lines',
                            name='RSI',
                            line=dict(color=self.colors['warning'], width=1)
                        ),
                        row=2, col=1
                    )
                    
                    # Add RSI overbought/oversold lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=df['MACD'],
                            mode='lines',
                            name='MACD',
                            line=dict(color=self.colors['primary'], width=1)
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=df['MACD_Signal'],
                            mode='lines',
                            name='MACD Signal',
                            line=dict(color=self.colors['secondary'], width=1)
                        ),
                        row=2, col=1
                    )
            
            # Add volume subplot
            if show_volume:
                volume_row = 3 if show_indicators else 2
                
                # Color volume bars based on price direction
                colors = ['red' if close < open else 'green' 
                         for close, open in zip(df['Close'], df['Open'])]
                
                fig.add_trace(
                    go.Bar(
                        x=df['Date'],
                        y=df['Volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=volume_row, col=1
                )
            
            # Update layout
            title = f"{symbol} Price Chart" if symbol else "Price Chart"
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white",
                height=600,
                showlegend=True,
                hovermode='x unified'
            )
            
            # Update x-axis
            fig.update_xaxes(
                rangeslider_visible=False,
                type='date'
            )
            
            logger.info("Price chart created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price chart: {str(e)}")
            return go.Figure()
    
    def create_trend_analysis_chart(self, df, analysis_results):
        """
        Create a trend analysis visualization
        
        Args:
            df (pd.DataFrame): Dataframe with price data
            analysis_results (dict): Results from trend analysis
            
        Returns:
            plotly.graph_objects.Figure: Trend analysis chart
        """
        try:
            logger.info("Creating trend analysis chart")
            
            if df.empty or not analysis_results:
                logger.warning("Empty data or analysis results for trend chart")
                return go.Figure()
            
            # Create subplots for different trend metrics
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=('Price with Trend Line', 'Trend Strength', 'Momentum', 'Support/Resistance'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Price with trend line
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color=self.colors['primary'], width=2)
                ),
                row=1, col=1
            )
            
            # Add trend line if available
            if 'trend_analysis' in analysis_results:
                trend = analysis_results['trend_analysis']
                if 'linear_trend_slope' in trend:
                    # Create trend line
                    x_trend = np.array([0, len(df)-1])
                    y_trend = trend['linear_trend_slope'] * x_trend + trend.get('intercept', 0)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'].iloc[[0, -1]],
                            y=y_trend,
                            mode='lines',
                            name='Trend Line',
                            line=dict(color=self.colors['danger'], width=2, dash='dash')
                        ),
                        row=1, col=1
                    )
            
            # Trend strength over time
            if 'trend_analysis' in analysis_results:
                trend = analysis_results['trend_analysis']
                if 'trend_strength' in trend:
                    fig.add_trace(
                        go.Indicator(
                            mode="gauge+number+delta",
                            value=trend['trend_strength'] * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Trend Strength (%)"},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': self.colors['primary']},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 80
                                }
                            }
                        ),
                        row=1, col=2
                    )
            
            # Momentum indicators
            if 'trend_analysis' in analysis_results:
                trend = analysis_results['trend_analysis']
                momentum_data = []
                labels = []
                
                if 'momentum_5d' in trend:
                    momentum_data.append(trend['momentum_5d'])
                    labels.append('5-Day')
                
                if 'momentum_20d' in trend:
                    momentum_data.append(trend['momentum_20d'])
                    labels.append('20-Day')
                
                if momentum_data:
                    fig.add_trace(
                        go.Bar(
                            x=labels,
                            y=momentum_data,
                            name='Momentum',
                            marker_color=[self.colors['success'] if m > 0 else self.colors['danger'] for m in momentum_data]
                        ),
                        row=2, col=1
                    )
            
            # Support and Resistance levels
            if 'trend_analysis' in analysis_results:
                trend = analysis_results['trend_analysis']
                if 'support_level' in trend and 'resistance_level' in trend:
                    current_price = df['Close'].iloc[-1]
                    
                    levels = [
                        {'name': 'Support', 'level': trend['support_level'], 'color': self.colors['success']},
                        {'name': 'Current Price', 'level': current_price, 'color': self.colors['primary']},
                        {'name': 'Resistance', 'level': trend['resistance_level'], 'color': self.colors['danger']}
                    ]
                    
                    for level in levels:
                        fig.add_trace(
                            go.Scatter(
                                x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
                                y=[level['level'], level['level']],
                                mode='lines',
                                name=level['name'],
                                line=dict(color=level['color'], width=2, dash='dash')
                            ),
                            row=2, col=2
                        )
            
            # Update layout
            fig.update_layout(
                title="Trend Analysis Dashboard",
                template="plotly_white",
                height=700,
                showlegend=True
            )
            
            logger.info("Trend analysis chart created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating trend analysis chart: {str(e)}")
            return go.Figure()
    
    def create_volatility_chart(self, df, volatility_results):
        """
        Create a volatility analysis visualization
        
        Args:
            df (pd.DataFrame): Dataframe with price data
            volatility_results (dict): Results from volatility analysis
            
        Returns:
            plotly.graph_objects.Figure: Volatility analysis chart
        """
        try:
            logger.info("Creating volatility chart")
            
            if df.empty or not volatility_results:
                logger.warning("Empty data or volatility results for volatility chart")
                return go.Figure()
            
            # Create subplots for volatility metrics
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=('Price Returns', 'Rolling Volatility', 'Volume Analysis', 'Risk Metrics'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Calculate returns
            returns = df['Close'].pct_change().dropna()
            
            # Price returns
            fig.add_trace(
                go.Scatter(
                    x=df['Date'].iloc[1:],
                    y=returns,
                    mode='lines',
                    name='Daily Returns',
                    line=dict(color=self.colors['primary'], width=1)
                ),
                row=1, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
            
            # Rolling volatility
            if len(returns) >= 20:
                rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'].iloc[20:],
                        y=rolling_vol,
                        mode='lines',
                        name='20-Day Rolling Volatility',
                        line=dict(color=self.colors['warning'], width=2)
                    ),
                    row=1, col=2
                )
            
            # Volume analysis
            if 'Volume' in df.columns:
                volume_sma = df['Volume'].rolling(window=20).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df['Volume'],
                        mode='lines',
                        name='Volume',
                        line=dict(color=self.colors['info'], width=1, opacity=0.7)
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=volume_sma,
                        mode='lines',
                        name='Volume SMA 20',
                        line=dict(color=self.colors['secondary'], width=2)
                    ),
                    row=2, col=1
                )
            
            # Risk metrics
            if volatility_results:
                risk_metrics = []
                metric_names = []
                
                if 'daily_volatility' in volatility_results:
                    risk_metrics.append(volatility_results['daily_volatility'] * 100)
                    metric_names.append('Daily Volatility (%)')
                
                if 'max_drawdown' in volatility_results:
                    risk_metrics.append(volatility_results['max_drawdown'] * 100)
                    metric_names.append('Max Drawdown (%)')
                
                if 'var_95' in volatility_results:
                    risk_metrics.append(volatility_results['var_95'] * 100)
                    metric_names.append('VaR 95% (%)')
                
                if risk_metrics:
                    fig.add_trace(
                        go.Bar(
                            x=metric_names,
                            y=risk_metrics,
                            name='Risk Metrics',
                            marker_color=[self.colors['danger'] if m < 0 else self.colors['success'] for m in risk_metrics]
                        ),
                        row=2, col=2
                    )
            
            # Update layout
            fig.update_layout(
                title="Volatility Analysis Dashboard",
                template="plotly_white",
                height=700,
                showlegend=True
            )
            
            logger.info("Volatility chart created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating volatility chart: {str(e)}")
            return go.Figure()
    
    def create_correlation_heatmap(self, correlation_matrix):
        """
        Create a correlation heatmap
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            
        Returns:
            plotly.graph_objects.Figure: Correlation heatmap
        """
        try:
            logger.info("Creating correlation heatmap")
            
            if correlation_matrix.empty:
                logger.warning("Empty correlation matrix for heatmap")
                return go.Figure()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(correlation_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Feature Correlation Heatmap",
                template="plotly_white",
                height=600,
                xaxis_title="Features",
                yaxis_title="Features"
            )
            
            logger.info("Correlation heatmap created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            return go.Figure()
    
    def create_market_regime_chart(self, regime_results):
        """
        Create a market regime visualization
        
        Args:
            regime_results (dict): Results from regime analysis
            
        Returns:
            plotly.graph_objects.Figure: Market regime chart
        """
        try:
            logger.info("Creating market regime chart")
            
            if not regime_results or 'regimes' not in regime_results:
                logger.warning("Empty regime results for regime chart")
                return go.Figure()
            
            regimes = regime_results['regimes']
            
            # Create 3D scatter plot for regime visualization
            fig = go.Figure()
            
            for regime_name, regime_data in regimes.items():
                fig.add_trace(
                    go.Scatter3d(
                        x=[regime_data['avg_return']],
                        y=[regime_data['avg_volatility']],
                        z=[regime_data['avg_momentum']],
                        mode='markers+text',
                        name=regime_name,
                        text=[regime_name],
                        textposition="middle center",
                        marker=dict(
                            size=regime_data['size'] / 10,  # Size based on regime size
                            color=regime_data['avg_return'],
                            colorscale='RdBu',
                            showscale=True
                        )
                    )
                )
            
            # Add current regime if available
            if 'current_regime' in regime_results and regime_results['current_regime']:
                current_regime = regimes[regime_results['current_regime']]
                fig.add_trace(
                    go.Scatter3d(
                        x=[current_regime['avg_return']],
                        y=[current_regime['avg_volatility']],
                        z=[current_regime['avg_momentum']],
                        mode='markers',
                        name='Current Regime',
                        marker=dict(
                            size=20,
                            color='red',
                            symbol='diamond'
                        )
                    )
                )
            
            fig.update_layout(
                title="Market Regime Analysis",
                template="plotly_white",
                height=600,
                scene=dict(
                    xaxis_title="Average Returns",
                    yaxis_title="Average Volatility",
                    zaxis_title="Average Momentum"
                )
            )
            
            logger.info("Market regime chart created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating market regime chart: {str(e)}")
            return go.Figure()
    
    def create_summary_dashboard(self, df, insights):
        """
        Create a comprehensive summary dashboard
        
        Args:
            df (pd.DataFrame): Dataframe with market data
            insights (dict): Comprehensive market insights
            
        Returns:
            plotly.graph_objects.Figure: Summary dashboard
        """
        try:
            logger.info("Creating summary dashboard")
            
            if df.empty or not insights:
                logger.warning("Empty data or insights for summary dashboard")
                return go.Figure()
            
            # Create subplots for different sections
            fig = sp.make_subplots(
                rows=3, cols=2,
                subplot_titles=('Price Performance', 'Market Sentiment', 'Risk Assessment', 'Technical Indicators', 'Volume Analysis', 'Summary Metrics'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Price performance
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color=self.colors['primary'], width=2)
                ),
                row=1, col=1
            )
            
            # Market sentiment gauge
            if 'summary' in insights and 'overall_market_sentiment' in insights['summary']:
                sentiment = insights['summary']['overall_market_sentiment']
                sentiment_value = {'Bullish': 100, 'Bearish': 0, 'Neutral': 50}.get(sentiment, 50)
                
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=sentiment_value,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Market Sentiment"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': self.colors['primary']},
                            'steps': [
                                {'range': [0, 30], 'color': "red"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ),
                    row=1, col=2
                )
            
            # Risk assessment
            if 'volatility_analysis' in insights and 'daily_volatility' in insights['volatility_analysis']:
                vol = insights['volatility_analysis']['daily_volatility']
                risk_level = "High" if vol > 0.3 else "Low" if vol < 0.15 else "Medium"
                risk_color = {"High": "red", "Medium": "yellow", "Low": "green"}.get(risk_level, "gray")
                
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=vol * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"Risk Level: {risk_level}"},
                        gauge={
                            'axis': {'range': [None, 50]},
                            'bar': {'color': risk_color},
                            'steps': [
                                {'range': [0, 15], 'color': "green"},
                                {'range': [15, 30], 'color': "yellow"},
                                {'range': [30, 50], 'color': "red"}
                            ]
                        }
                    ),
                    row=2, col=1
                )
            
            # Technical indicators summary
            if 'trend_analysis' in insights:
                trend = insights['trend_analysis']
                indicators = []
                values = []
                
                if 'trend_direction' in trend:
                    indicators.append('Trend Direction')
                    values.append(trend['trend_direction'])
                
                if 'trend_strength' in trend:
                    indicators.append('Trend Strength')
                    values.append(f"{trend['trend_strength']:.2f}")
                
                if indicators:
                    fig.add_trace(
                        go.Table(
                            header=dict(values=['Indicator', 'Value']),
                            cells=dict(values=[indicators, values])
                        ),
                        row=2, col=2
                    )
            
            # Volume analysis
            if 'Volume' in df.columns:
                recent_volume = df['Volume'].tail(20).mean()
                historical_volume = df['Volume'].mean()
                volume_change = (recent_volume - historical_volume) / historical_volume * 100
                
                fig.add_trace(
                    go.Indicator(
                        mode="number+delta",
                        value=recent_volume,
                        delta={'reference': historical_volume, 'relative': True},
                        title={'text': "Recent vs Historical Volume"},
                        number={'valueformat': ',.0f'}
                    ),
                    row=3, col=1
                )
            
            # Summary metrics table
            if 'summary' in insights and 'key_insights' in insights['summary']:
                key_insights = insights['summary']['key_insights']
                fig.add_trace(
                    go.Table(
                        header=dict(values=['Key Insights']),
                        cells=dict(values=[key_insights])
                    ),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="Market Trend Analysis Dashboard",
                template="plotly_white",
                height=1000,
                showlegend=False
            )
            
            logger.info("Summary dashboard created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating summary dashboard: {str(e)}")
            return go.Figure()

def main():
    """Test function for the visualization module"""
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
    
    # Add some technical indicators
    sample_data['SMA_20'] = sample_data['Close'].rolling(window=20).mean()
    sample_data['SMA_50'] = sample_data['Close'].rolling(window=50).mean()
    sample_data['RSI'] = 50 + np.random.randn(100) * 20
    
    visualizer = MarketVisualizer()
    
    # Test price chart
    print("Testing price chart creation...")
    price_chart = visualizer.create_price_chart(sample_data, 'AAPL')
    print(f"Price chart created: {type(price_chart)}")
    
    # Test trend analysis chart
    print("\nTesting trend analysis chart...")
    trend_results = {'trend_analysis': {'linear_trend_slope': 0.1, 'trend_strength': 0.7}}
    trend_chart = visualizer.create_trend_analysis_chart(sample_data, trend_results)
    print(f"Trend chart created: {type(trend_chart)}")
    
    # Test volatility chart
    print("\nTesting volatility chart...")
    volatility_results = {'daily_volatility': 0.2, 'max_drawdown': -0.1}
    volatility_chart = visualizer.create_volatility_chart(sample_data, volatility_results)
    print(f"Volatility chart created: {type(volatility_chart)}")
    
    # Test correlation heatmap
    print("\nTesting correlation heatmap...")
    correlation_matrix = sample_data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    correlation_chart = visualizer.create_correlation_heatmap(correlation_matrix)
    print(f"Correlation chart created: {type(correlation_chart)}")

if __name__ == "__main__":
    main()


