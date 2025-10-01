Walmart Sales Forecasting Project
Project Overview
This project predicts future sales for Walmart stores using machine learning. It helps Walmart plan inventory, manage staff, and make better business decisions.

Business Problem
Walmart needs to forecast sales to:

Avoid running out of stock
Prevent overstocking products
Plan staff schedules during busy periods

Make accurate financial predictions

Dataset
We use Walmart's sales data with information about:
Store details (size, type)
Weekly sales numbers
Holiday dates
Economic factors (fuel prices, unemployment)

What This Project Does
1 Data Preparation
Combines multiple data files
Handles missing values
Adds useful features like month and week numbers

2 Data Analysis
Shows sales trends over time
Compares store performance
Analyzes holiday impact on sales
Finds seasonal patterns

3 Sales Forecasting
We test three different models:

ARIMA - Traditional time series model
RMSE: $7,474.87
Good for basic patterns
SARIMA - Advanced version that understands seasons
RMSE: $50,622.28
Handles yearly patterns
Prophet - Facebook's modern forecasting tool
RMSE: $5,897.63 Best Performance
Great with holidays and seasons

Future Predictions
The model can predict sales 12 weeks into the future with confidence intervals.

Key Results
Model Performance
Model	RMSE	MAE	MAPE
ARIMA	$7,475	$6,642	-
SARIMA	$50,622	$50,157	-
Prophet	$5,898	$3,868	Best

Business Insights
Holidays boost sales by significant amounts

Store Type A performs best
Certain months have consistently higher sales
Some departments sell much more than others
