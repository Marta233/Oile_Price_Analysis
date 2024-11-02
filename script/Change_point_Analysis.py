import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import ruptures as rpt  # Import the ruptures library for change point detection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChangePointAnalysis:
    def __init__(self, data, events_json=None):
        """Initialize the ChangePointAnalysis class and load the data."""
        logging.info("Loading data.")
        self.data = data
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        self.model_fit = None
        self.events = []
        
        # Load events if a JSON file is provided
        if events_json:
            self.load_events(events_json)

        logging.info("Data loaded successfully. Number of records: %d", len(self.data))

    def load_events(self, events_json):
        """Load events from a JSON file."""
        logging.info("Loading events from JSON file.")
        with open(events_json, 'r') as f:
            events_data = json.load(f)
            self.events = events_data.get('events', [])
            logging.info("Loaded %d events.", len(self.events))

    def visualize_data(self):
        """Visualize the time series data."""
        logging.info("Visualizing the time series data.")
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Price'], label='Price', color='blue')
        plt.title('Brent Oil Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def cusum_analysis(self):
        """Perform CUSUM change point analysis."""
        logging.info("Performing CUSUM change point analysis.")
        if 'Date' in self.data.columns:
            self.data['Year'] = pd.to_datetime(self.data['Date']).dt.year
        else:
            logging.warning("DataFrame does not have a 'Date' column. Assuming the index is the date.")
            self.data['Year'] = self.data.index.year
            
        price = self.data['Price'].values
        year = self.data['Year'].values
        mean_price = np.mean(price)
        cusum = np.cumsum(price - mean_price)
        threshold = 3 * np.std(price)  # You can adjust this threshold as needed
        change_points = np.where(np.abs(cusum) > threshold)[0]
        change_years = year[change_points]
        
        # Visualize CUSUM results
        plt.figure(figsize=(12, 6))
        plt.plot(year, cusum, label='CUSUM', color='orange')
        plt.axhline(threshold, linestyle='--', color='red', label='Upper Threshold')
        plt.axhline(-threshold, linestyle='--', color='green', label='Lower Threshold')
        plt.title('CUSUM Change Point Analysis')
        plt.xlabel('Year')
        plt.ylabel('Cumulative Sum')
        plt.legend()
        plt.show()

        logging.info("Detected change points at years: %s", change_years)

    def change_point_detection(self):
        """Perform change point detection using PELT algorithm."""
        logging.info("Performing change point detection using PELT algorithm.")
        price = self.data['Price'].values

        # Using PELT algorithm for change point detection
        algo = rpt.Pelt(model="rbf").fit(price)
        change_points = algo.predict(pen=5)  # Adjust the penalty term as needed

        # Visualize change point results with events
        plt.figure(figsize=(14, 6))
        plt.plot(self.data.index, self.data['Price'], label='Price', color='blue')

        for cp in change_points[:-1]:
            plt.axvline(x=self.data.index[cp], color='red', linestyle='--', label='Change Point' if cp==change_points[0] else "")
        
        # Plot events
        self._plot_events()

        plt.title('Change Point Detection using PELT with Events')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

        logging.info("Detected change points at: %s", change_points)

    def _plot_events(self):
        """Plot events on the time series graph."""
        for event in self.events:
            if 'year' in event and 'event' in event:
                event_date = pd.to_datetime(event['year'])
                plt.axvline(x=event_date, color='purple', linestyle='--', label=event['event'])
                plt.text(event_date, plt.ylim()[1]*0.9, event['event'], rotation=45, color='purple')
            else:
                logging.warning("Event missing required fields: %s", event)


