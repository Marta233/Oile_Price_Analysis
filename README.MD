# Forecasting Brent Oil Prices: Analyzing Historical Trends and Economic Influences 

## Overview
This repository contains the code and documentation for a project aimed at analyzing the historical price trends of Brent crude oil and identifying the key economic factors that influence its fluctuations. The goal is to develop a robust forecasting model that can help stakeholders make more informed decisions and navigate the volatility of the oil market.

## Data
The dataset consists of:
- **Price information**: daily Brent crude oil prices from 1988 to 2024
- **Other Data**:  macroeconomic indicators such as global GDP, inflation rates, and geopolitical events. 

## Methodology
1. **Data Preprocessing**: Cleaning, normalization, and transformation to prepare raw data for analysis.
2. **Feature Engineering**: Development of domain-relevant features that may indicate fraud patterns.
3. **Model Training**: Training a Random Forest classifier and evaluating its performance using metrics like accuracy, precision, and recall.
4. **Model Interpretability**: Applying LIME and SHAP to understand feature importance and enhance transparency in the decision-making process.

## Deployment
The trained model is packaged and deployed as a web service, enabling integration with the organization’s systems for real-time fraud detection.

## Installation and Usage
[Please refer to the project documentation for setup instructions and usage guidelines for the fraud detection model.](https://github.com/Marta233/Oile_Price_Analysis.git)
### 1. Clone the Repository
To get started, clone the project repository to your local machine:
```bash
git clone https://github.com/Marta233/Oile_Price_Analysis.git
```
cd Oile_Price_Analysis
### 2. Set Up a Virtual Environment (venv)
Create a virtual environment to isolate the project dependencies:
```bash
python3 -m my_venv .venv
```
```bash
source .my_venv/Scripts/activate  
```
```bash
pip install -r requirements.txt
```
## Contributing
Contributions are welcome! Please refer to the repository's guidelines for submitting bug reports, feature requests, and pull requests.
