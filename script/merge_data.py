import pandas as pd
import json

class EventPriceIntegrator:
    def __init__(self, events_json, prices_data):
        # Load the events data from the JSON file
        with open(events_json, 'r') as f:
            data = json.load(f)
        
        # Extract the events into a DataFrame
        self.events_df = pd.DataFrame(data['events'])
        
        # Ensure date columns are in datetime format
        self.events_df['start_date'] = pd.to_datetime(self.events_df['start_date'])
        self.events_df['end_date'] = pd.to_datetime(self.events_df['end_date'])

        print("Loaded Events DataFrame:")
        print(self.events_df)

        # Load the prices data
        self.prices_df = pd.read_csv(prices_data)
        self.prices_df['Date'] = pd.to_datetime(self.prices_df['Date'], format='mixed')

        print("Loaded Prices DataFrame:")
        print(self.prices_df)

    def integrate_events(self):
        """Integrate event columns into the price DataFrame."""
        for _, row in self.events_df.iterrows():
            event_name = row['event']
            print(f"Processing event: {event_name} from {row['start_date']} to {row['end_date']}")
            
            mask = (self.prices_df['Date'] >= row['start_date']) & (self.prices_df['Date'] <= row['end_date'])
            self.prices_df[event_name] = mask.astype(int)
            print(f"Matched {mask.sum()} dates for event: {event_name}")

    def get_result(self):
        """Return the integrated DataFrame."""
        return self.prices_df

