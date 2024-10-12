# USA Rain Prediction App (2024 - 2025)

This project aims to predict whether it will rain tomorrow in major US cities based on weather data from 2024-2025.

You can view the app here: [USA Rain Prediction](https://usa-rain-prediction.streamlit.app/)

## Overview
- Predicts next-day rain (yes/no) for 20 major US cities
- Uses weather data from 2024-2025 including temperature, humidity, wind speed, etc.
- Implements multiple machine learning models with hyperparameter tuning
- Achieves 100% accuracy with optimized Random Forest classifier
## Dataset
The dataset contains daily weather measurements for 20 US cities over 2 years, including:

- Date
- Location
- Temperature
- Humidity
- Wind Speed
- Precipitation
- Cloud Cover
- Atmospheric Pressure
- Rain Tomorrow (target variable)
## Models
Several models were evaluated:
- Random Forest
- K-Nearest Neighbors
- XGBoost
`Random Forest` performed best after hyperparameter optimization.

## Results
The final Random Forest model achieved:
- 100% accuracy
- 1.00 precision, recall and F1-score for both classes

## Usage

Install required packages:
```bash
pip install -r requirements.txt
```
