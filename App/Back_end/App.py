from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from io import StringIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your Brent oil price data
data = pd.read_csv('data/BrentOilPrices.csv')  # Adjust the path as necessary

data_event = pd.read_csv('data/merged_oil_price_history.csv')  # Adjust the path as necessary


@app.route('/api/prices', methods=['GET'])
def get_prices():
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/events', methods=['GET'])
def get_events():
    return jsonify(data_event.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
