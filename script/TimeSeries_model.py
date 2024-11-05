import wbdata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR, ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from arch import arch_model  # GARCH model import
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

class TimeSeriesModeling:
    def __init__(self, merged_data):
        self.merged_data = merged_data
        self.scaled_data = None
        self.time_step = 10  # Number of previous days to use for prediction
        self.model_results = {}
        self.metrics = {}

    def preprocess_data(self):
        """Preprocess the merged data."""
        self.merged_data.dropna(inplace=True)  # Drop rows with missing values
        print("Data preprocessed. Rows after cleaning:", len(self.merged_data))

    def plot_time_series(self):
        """Plot the time series data."""
        plt.figure(figsize=(12, 8))
        for column in self.merged_data.columns:
            plt.plot(self.merged_data.index, self.merged_data[column], label=column)
        plt.title("Time Series Data Overview")
        plt.xlabel("Date")
        plt.ylabel("Values")
        plt.legend()
        plt.show()

    def check_stationarity(self, column):
        """Perform the Augmented Dickey-Fuller test to check stationarity."""
        result = adfuller(self.merged_data[column].dropna())
        print(f"Augmented Dickey-Fuller test for {column}:")
        print(f"Test Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        print(f"Critical Values: {result[4]}")
        return result[1] <= 0.05  # Return True if stationary

    def difference_data(self, column):
        """Apply differencing to make the data stationary."""
        diff_data = self.merged_data[column].diff().dropna()
        return diff_data

    def fit_var_model(self):
        """Fit a VAR model, forecast, and plot aggregated results."""
        model_data = self.merged_data.copy().dropna()
        
        # Check stationarity and difference if necessary
        if not self.check_stationarity(model_data.columns[0]):  # Assume first column for VAR
            print("Differencing the data for stationarity.")
            model_data = model_data.diff().dropna()

        model = VAR(model_data)
        results = model.fit(maxlags=5, ic='aic')
        self.model_results['VAR'] = results

        forecast = results.forecast(model_data.values[-results.k_ar:], steps=10)
        forecast_index = pd.date_range(start=model_data.index[-1] + pd.Timedelta(days=1), periods=10, freq='D')
        forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=model_data.columns)
        
        historical_avg = model_data.mean(axis=1)
        forecast_avg = forecast_df.mean(axis=1)
        combined_avg = pd.concat([historical_avg, forecast_avg])

        plt.figure(figsize=(12, 6))
        plt.plot(historical_avg.index, historical_avg, label="Historical Average Price", color='blue')
        plt.plot(forecast_avg.index, forecast_avg, label="Forecasted Average Price", linestyle='--', color='red')
        plt.title("VAR Model - Historical and Forecasted Average Price")
        plt.xlabel("Date")
        plt.ylabel("Average Price")
        plt.legend(loc='upper left')
        plt.show()

        print("VAR model fitted, forecasted, and plotted (average values).")
        self.calculate_metrics(historical_avg[-10:], forecast_avg, "VAR")
        return forecast_df
    
    def fit_arima_model(self, column='Price'):
        """Fit an ARIMA model for a specific column and plot the forecast."""
        model_data = self.merged_data[column].dropna()
        
        # Check stationarity and difference if necessary
        if not self.check_stationarity(column):
            print(f"Differencing {column} for stationarity.")
            model_data = self.difference_data(column)

        model = ARIMA(model_data, order=(5, 1, 0))
        results = model.fit()
        self.model_results['ARIMA'] = results

        forecast = results.get_forecast(steps=10)
        forecast_df = forecast.summary_frame()
        forecast_index = pd.date_range(start=model_data.index[-1] + pd.Timedelta(days=1), periods=10, freq='D')
        forecast_df.index = forecast_index

        plt.figure(figsize=(12, 6))
        plt.plot(model_data.index, model_data, label="Historical Data", color='blue')
        plt.plot(forecast_df.index, forecast_df['mean'], label="Forecasted Data", linestyle='--', color='red')
        plt.fill_between(forecast_df.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='pink', alpha=0.3)
        plt.title(f"ARIMA Model - {column} Forecast")
        plt.xlabel("Date")
        plt.ylabel(column)
        plt.legend(loc='upper left')
        plt.show()

        print("ARIMA model fitted, forecasted, and plotted.")
        self.calculate_metrics(model_data[-10:], forecast_df['mean'], "ARIMA")
        return forecast_df

    def fit_garch_model(self, column='Price'):
        """Fit a GARCH model for volatility prediction and plot the forecast."""
        model_data = self.merged_data[column].dropna()
        
        # Check stationarity and difference if necessary
        if not self.check_stationarity(column):
            print(f"Differencing {column} for stationarity.")
            model_data = self.difference_data(column)

        model = arch_model(model_data, vol='Garch', p=1, q=1)
        results = model.fit(disp='off')
        self.model_results['GARCH'] = results

        forecast = results.forecast(horizon=10)
        forecast_variance = forecast.variance.values[-1, :]

        plt.figure(figsize=(12, 6))
        plt.plot(model_data.index, model_data, label="Historical Data", color='blue')
        plt.plot(pd.date_range(start=model_data.index[-1] + pd.Timedelta(days=1), periods=10, freq='D'), forecast_variance,
                 label="Forecasted Volatility", linestyle='--', color='red')
        plt.title(f"GARCH Model - {column} Forecasted Volatility")
        plt.xlabel("Date")
        plt.ylabel("Variance")
        plt.legend(loc='upper left')
        plt.show()

        print("GARCH model fitted, forecasted, and plotted.")
        return forecast_variance
    def prepare_lstm_data(self):
        """Prepare data for LSTM with the specified time step."""
        data = self.scaled_data if self.scaled_data is not None else self.merged_data.values
        X, y = [], []
        for i in range(len(data) - self.time_step):
            X.append(data[i:i + self.time_step])
            y.append(data[i + self.time_step])  # Next step prediction as target
        X, y = np.array(X), np.array(y)
        return X, y.reshape(-1, 1)  # Reshape y to ensure consistent shape

    def fit_lstm_model(self, column='Price'):
        """Fit an LSTM model for time series forecasting."""
        # Preprocess data for LSTM
        self.scaled_data = MinMaxScaler().fit_transform(self.merged_data[[column]])
        X, y = self.prepare_lstm_data()  # Prepare sequences
        
        # Reshape y to match samples for consistent dimensions
        if X.shape[0] != y.shape[0]:
            y = y[:X.shape[0]]
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
        
        # Store and plot results
        self.model_results['LSTM'] = model
        predictions = model.predict(X)
        self.calculate_metrics(y, predictions, "LSTM")  # Save metrics
        self.plot_lstm_results(X, y, predictions)
        
    def plot_lstm_results(self, X, y, predictions):
        """Plot historical data and LSTM predictions."""
        plt.figure(figsize=(12, 6))
        plt.plot(y, label="Actual", color="blue")
        plt.plot(predictions, label="Predicted", color="red", linestyle="--")
        plt.title("LSTM Model - Actual vs. Predicted")
        plt.xlabel("Time Step")
        plt.ylabel("Price")
        plt.legend(loc="upper left")
        plt.show()

    def fit_markov_model(self, column='Price'):
        """Fit a Markov Model for predicting regime changes within the sample range."""
        model_data = self.merged_data[column].dropna()
        
        # Check stationarity and difference if necessary
        if not self.check_stationarity(column):
            print(f"Differencing {column} for stationarity.")
            model_data = self.difference_data(column)

        model = MarkovRegression(model_data, k_regimes=2)
        results = model.fit()
        self.model_results['Markov'] = results

        # In-sample predictions (within the available data range)
        in_sample_preds = results.predict()

        plt.figure(figsize=(12, 6))
        plt.plot(model_data.index, model_data, label="Historical Data", color='blue')
        plt.plot(model_data.index, in_sample_preds, label="In-Sample Predictions", linestyle='--', color='red')
        plt.title(f"Markov Model - {column} In-Sample Prediction")
        plt.xlabel("Date")
        plt.ylabel(column)
        plt.legend(loc='upper left')
        plt.show()

        print("Markov model fitted and in-sample predictions plotted.")
        self.calculate_metrics(model_data, in_sample_preds, "Markov")
        return in_sample_preds


    def calculate_metrics(self, actual, predicted, model_name):
        """Calculate metrics for the model performance."""
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        self.metrics[model_name] = {
            'MSE': mse,
            'MAE': mae
        }
        print(f"{model_name} - MSE: {mse}, MAE: {mae}")

    def save_metrics_to_csv(self, file_name='model_performance_metrics.csv'):
        """Save the performance metrics to a CSV file."""
        metrics_df = pd.DataFrame.from_dict(self.metrics, orient='index')
        metrics_df.to_csv(file_name)
        print(f"Performance metrics saved to {file_name}")

