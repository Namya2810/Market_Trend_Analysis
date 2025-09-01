"""
Machine Learning Model Module for Market Trend Analysis
Handles model training, predictions, and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketPredictor:
    """Class for market prediction using machine learning models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
    def train_regression_model(self, X, y, model_type='random_forest', test_size=0.2, random_state=42):
        """
        Train a regression model for price prediction
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target variable (price changes)
            model_type (str): Type of model to train
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Model performance metrics
        """
        try:
            logger.info(f"Training {model_type} regression model")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select and train model
            if model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=random_state)
            elif model_type == 'linear_regression':
                model = LinearRegression()
            elif model_type == 'svr':
                model = SVR(kernel='rbf')
            elif model_type == 'neural_network':
                model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=random_state, max_iter=500)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store model and scaler
            model_key = f"{model_type}_regression"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_key] = model.feature_importances_
            
            # Store performance metrics
            performance = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse),
                'test_size': len(X_test),
                'train_size': len(X_train)
            }
            
            self.model_performance[model_key] = performance
            
            logger.info(f"Model training completed. R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
            return performance
            
        except Exception as e:
            logger.error(f"Error training regression model: {str(e)}")
            return {}
    
    def train_classification_model(self, X, y, model_type='random_forest', test_size=0.2, random_state=42):
        """
        Train a classification model for trend prediction
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target variable (trend labels: 1 for up, 0 for down)
            model_type (str): Type of model to train
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Model performance metrics
        """
        try:
            logger.info(f"Training {model_type} classification model")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select and train model
            if model_type == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            elif model_type == 'logistic_regression':
                model = LogisticRegression(random_state=random_state)
            elif model_type == 'svc':
                model = SVC(kernel='rbf', random_state=random_state)
            elif model_type == 'neural_network':
                model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=random_state, max_iter=500)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store model and scaler
            model_key = f"{model_type}_classification"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_key] = model.feature_importances_
            
            # Store performance metrics
            performance = {
                'accuracy': accuracy,
                'test_size': len(X_test),
                'train_size': len(X_train),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            self.model_performance[model_key] = performance
            
            logger.info(f"Model training completed. Accuracy: {accuracy:.4f}")
            return performance
            
        except Exception as e:
            logger.error(f"Error training classification model: {str(e)}")
            return {}
    
    def hyperparameter_tuning(self, X, y, model_type='random_forest', cv=5, random_state=42):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target variable
            model_type (str): Type of model to tune
            cv (int): Number of cross-validation folds
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Best parameters and performance
        """
        try:
            logger.info(f"Performing hyperparameter tuning for {model_type}")
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Define parameter grids
            if model_type == 'random_forest':
                model = RandomForestRegressor(random_state=random_state)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_type == 'svr':
                model = SVR()
                param_grid = {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'kernel': ['rbf', 'linear']
                }
            elif model_type == 'neural_network':
                model = MLPRegressor(random_state=random_state, max_iter=500)
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            else:
                logger.warning(f"Hyperparameter tuning not implemented for {model_type}")
                return {}
            
            # Perform grid search
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring='neg_mean_squared_error',
                n_jobs=-1, random_state=random_state
            )
            
            grid_search.fit(X_scaled, y)
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_  # Convert back to positive MSE
            
            # Store best model
            model_key = f"{model_type}_tuned"
            self.models[model_key] = best_model
            self.scalers[model_key] = scaler
            
            # Store performance
            performance = {
                'best_params': best_params,
                'best_mse': best_score,
                'best_rmse': np.sqrt(best_score),
                'cv_folds': cv
            }
            
            self.model_performance[model_key] = performance
            
            logger.info(f"Hyperparameter tuning completed. Best RMSE: {np.sqrt(best_score):.4f}")
            return performance
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            return {}
    
    def make_prediction(self, X, model_key=None):
        """
        Make predictions using a trained model
        
        Args:
            X (np.array): Feature matrix for prediction
            model_key (str): Key of the model to use
            
        Returns:
            np.array: Predictions
        """
        try:
            if not self.models:
                logger.error("No trained models available")
                return None
            
            if model_key is None:
                # Use the first available model
                model_key = list(self.models.keys())[0]
            
            if model_key not in self.models:
                logger.error(f"Model {model_key} not found")
                return None
            
            model = self.models[model_key]
            scaler = self.scalers.get(model_key)
            
            # Scale features if scaler is available
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            
            # Make prediction
            predictions = model.predict(X_scaled)
            
            logger.info(f"Predictions made using {model_key}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None
    
    def evaluate_model(self, X, y, model_key=None):
        """
        Evaluate model performance on new data
        
        Args:
            X (np.array): Feature matrix
            y (np.array): True target values
            model_key (str): Key of the model to evaluate
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            if not self.models:
                logger.error("No trained models available")
                return {}
            
            if model_key is None:
                # Use the first available model
                model_key = list(self.models.keys())[0]
            
            if model_key not in self.models:
                logger.error(f"Model {model_key} not found")
                return {}
            
            model = self.models[model_key]
            scaler = self.scalers.get(model_key)
            
            # Scale features if scaler is available
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            
            # Calculate metrics based on model type
            if 'regression' in model_key:
                metrics = {
                    'mse': mean_squared_error(y, y_pred),
                    'mae': mean_absolute_error(y, y_pred),
                    'r2': r2_score(y, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y, y_pred))
                }
            elif 'classification' in model_key:
                metrics = {
                    'accuracy': accuracy_score(y, y_pred),
                    'classification_report': classification_report(y, y_pred, output_dict=True)
                }
            else:
                metrics = {}
            
            logger.info(f"Model evaluation completed for {model_key}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {}
    
    def get_feature_importance(self, model_key=None, top_n=10):
        """
        Get feature importance from trained models
        
        Args:
            model_key (str): Key of the model to get importance from
            top_n (int): Number of top features to return
            
        Returns:
            dict: Feature importance scores
        """
        try:
            if not self.feature_importance:
                logger.warning("No feature importance available")
                return {}
            
            if model_key is None:
                # Use the first available model
                model_key = list(self.feature_importance.keys())[0]
            
            if model_key not in self.feature_importance:
                logger.error(f"Feature importance for {model_key} not found")
                return {}
            
            importance_scores = self.feature_importance[model_key]
            
            # Sort features by importance
            sorted_features = sorted(
                enumerate(importance_scores), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Return top N features
            top_features = {
                f"Feature_{i}": score for i, score in sorted_features[:top_n]
            }
            
            return top_features
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def save_model(self, model_key, filepath):
        """
        Save a trained model to disk
        
        Args:
            model_key (str): Key of the model to save
            filepath (str): Path to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if model_key not in self.models:
                logger.error(f"Model {model_key} not found")
                return False
            
            model = self.models[model_key]
            scaler = self.scalers.get(model_key)
            
            # Save model and scaler
            model_path = f"{filepath}_{model_key}.joblib"
            scaler_path = f"{filepath}_{model_key}_scaler.joblib"
            
            joblib.dump(model, model_path)
            if scaler is not None:
                joblib.dump(scaler, scaler_path)
            
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_key, filepath):
        """
        Load a trained model from disk
        
        Args:
            model_key (str): Key for the loaded model
            filepath (str): Path to load the model from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            model_path = f"{filepath}_{model_key}.joblib"
            scaler_path = f"{filepath}_{model_key}_scaler.joblib"
            
            # Load model
            model = joblib.load(model_path)
            self.models[model_key] = model
            
            # Load scaler if available
            try:
                scaler = joblib.load(scaler_path)
                self.scalers[model_key] = scaler
            except FileNotFoundError:
                logger.warning(f"Scaler not found for {model_key}")
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_model_summary(self):
        """
        Get a summary of all trained models
        
        Returns:
            dict: Summary of models and their performance
        """
        try:
            summary = {
                'total_models': len(self.models),
                'model_types': list(self.models.keys()),
                'performance': self.model_performance,
                'feature_importance_available': bool(self.feature_importance)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model summary: {str(e)}")
            return {}

def main():
    """Test function for the model module"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y_regression = np.random.randn(n_samples)  # For regression
    y_classification = (np.random.randn(n_samples) > 0).astype(int)  # For classification
    
    predictor = MarketPredictor()
    
    # Test regression model training
    print("Testing regression model training...")
    reg_performance = predictor.train_regression_model(X, y_regression, 'random_forest')
    print(f"Regression performance: {reg_performance}")
    
    # Test classification model training
    print("\nTesting classification model training...")
    clf_performance = predictor.train_classification_model(X, y_classification, 'random_forest')
    print(f"Classification performance: {clf_performance}")
    
    # Test hyperparameter tuning
    print("\nTesting hyperparameter tuning...")
    tune_performance = predictor.hyperparameter_tuning(X, y_regression, 'random_forest')
    print(f"Tuning performance: {tune_performance}")
    
    # Test predictions
    print("\nTesting predictions...")
    X_test = np.random.randn(100, n_features)
    predictions = predictor.make_prediction(X_test)
    print(f"Predictions shape: {predictions.shape if predictions is not None else 'None'}")
    
    # Test feature importance
    print("\nTesting feature importance...")
    importance = predictor.get_feature_importance()
    print(f"Top features: {importance}")
    
    # Test model summary
    print("\nTesting model summary...")
    summary = predictor.get_model_summary()
    print(f"Model summary: {summary}")

if __name__ == "__main__":
    main()


