import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeSeriesAnalysis:
    def __init__(self, data):
        """Initialize the TimeSeriesAnalysis class and load the data."""
        logging.info("Loading data %s")
        self.data=data
        self.model_fit = None
        logging.info("Data loaded successfully. Number of records: %d", len(self.data))

    def visualize_data(self):
        """Visualize the time series data."""
        logging.info("Visualizing the time series data.")
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Price'], label='Price', color='blue')
        plt.title('Brent Oil Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def check_stationarity(self):
        """Check if the time series is stationary using the ADF test."""
        logging.info("Checking for stationarity.")
        result = adfuller(self.data['price'])
        logging.info('ADF Statistic: %.4f', result[0])
        logging.info('p-value: %.4f', result[1])
        if result[1] < 0.05:
            logging.info("The time series is stationary.")
        else:
            logging.warning("The time series is not stationary. Consider differencing the data.")

    def differencing(self):
        """Apply differencing to the time series data."""
        logging.info("Applying differencing to the time series data.")
        self.data['price_diff'] = self.data['price'].diff()
        self.data['price_diff'].dropna(inplace=True)
        logging.info("Differencing applied. New data length: %d", len(self.data['price_diff'].dropna()))

    def plot_acf_pacf(self):
        """Plot the ACF and PACF to determine AR and MA terms."""
        logging.info("Plotting ACF and PACF.")
        plt.figure(figsize=(12, 6))
        plot_acf(self.data['price_diff'].dropna())
        plt.title('ACF Plot')
        plt.show()

        plt.figure(figsize=(12, 6))
        plot_pacf(self.data['price_diff'].dropna())
        plt.title('PACF Plot')
        plt.show()

    def fit_arima(self, order):
        """Fit the ARIMA model to the data."""
        logging.info("Fitting ARIMA model with order %s", order)
        model = ARIMA(self.data['price'], order=order)
        self.model_fit = model.fit()
        logging.info("Model fitted successfully.")
        print(self.model_fit.summary())

    def make_predictions(self, steps=30):
        """Make future predictions using the fitted ARIMA model."""
        if self.model_fit is not None:
            logging.info("Making predictions for %d steps ahead.", steps)
            forecast = self.model_fit.forecast(steps=steps)
            logging.info("Predictions made successfully.")
            return forecast
        else:
            logging.error("Model is not fitted yet. Please fit the model before making predictions.")
            return None

    def evaluate_model(self, actual_prices):
        """Evaluate the model using MAE and RMSE metrics."""
        if self.model_fit is not None:
            mae = mean_absolute_error(actual_prices, self.model_fit.fittedvalues)
            rmse = np.sqrt(mean_squared_error(actual_prices, self.model_fit.fittedvalues))
            logging.info('Model Evaluation: MAE: %.4f, RMSE: %.4f', mae, rmse)
        else:
            logging.error("Model is not fitted yet. Please fit the model before evaluation.")

