# Walmart Sales Forecasting Project

## Project Overview
This project predicts future sales for Walmart stores using machine learning. It helps Walmart plan inventory, manage staff, and make better business decisions.

## Business Problem
Walmart needs to forecast sales to:

* Avoid running out of stock
* Prevent overstocking products
* Plan staff schedules during busy periods
* Make accurate financial predictions

## Dataset
We use Walmart's sales data with information about:
* Store details (size, type)
* Weekly sales numbers
* Holiday dates
* Economic factors (fuel prices, unemployment)

## What This Project Does
### 1 Data Preparation
* Combines multiple data files
* Handles missing values
* Adds useful features like month and week numbers

### 2 Data Analysis
* Shows sales trends over time
* Compares store performance
* Analyzes holiday impact on sales
* Finds seasonal patterns

### 3 Sales Forecasting
We test three different models:

* **ARIMA** - Traditional time series model
  * RMSE: $7,474.87
  * Good for basic patterns
* **SARIMA** - Advanced version that understands seasons
  * RMSE: $50,622.28
  * Handles yearly patterns
* **Prophet** - Facebook's modern forecasting tool
  * RMSE: $5,897.63 Best Performance
  * Great with holidays and seasons

## Future Predictions
The model can predict sales 12 weeks into the future with confidence intervals.

## Key Results
## Model Performance
Model	RMSE	MAE	MAPE
ARIMA	$7,475	$6,642	-
SARIMA	$50,622	$50,157	-
Prophet	$5,898	$3,868	Best

## Business Insights

* **Sales Performance**
  * Total sales: $6.7 billion
  * Best week: $80.9 million in sales
  * Average week: $47.1 million in sales
* **Store Types**
  * Type A stores perform best ($20,100 weekly average)
  * Type B: $12,237 weekly average
  * Type C: $9,520 weekly average
* **Key Patterns**
  * Holidays boost sales by 7.1%
  * December is the best month for sales ($19,356 average)
  * January is the slowest month ($14,126 average)
* **What Affects Sales**
  * Economic factors like fuel prices and unemployment have very little impact on sales
  * Time patterns (holidays, seasons) are much better predictors
* **Best Forecasting Model**
  * Prophet model performed best being 21% more accurate than other models
  * Can reliably predict sales up to 12 weeks into the future
