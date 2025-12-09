# ForecastHub: Local Python Sales Forecasting Tool
A Python-based sales forecasting tool that runs entirely on your computer. No cloud connectivity required which means that no sensitive data will leave your machine. Also included: a Graphical User Interface that allows the user to both upload and download data without touching any code, thus offering a user-friendly and code-safe forecasting tool. Finally, in regards to the code, all file paths have been soft-coded, which means that as long as you keep all the components of the code in the same folder, it will run anywhere. All you need is Python installed on your computer.
![ForecastHub Interface](ForecastHub.jpeg)

## 🔧 How It Works (Technically)
Built with the below Python libraries:

Pandas loads and cleans your sales data

Statsmodels handles time series forecasting (ARIMA, Exponential Smoothing)

Scikit-learn provides machine learning components (Random Forest)

The forecast tool analyzes historical sales, detects patterns and trends, and generates forecasts with confidence intervals.

## 🖱️ GUI Interface attached (No Coding Needed)
Included within the code is a graphical interface permititng to upload and download data easily with only a few clicks.  

Import CSV/Excel files

Adjust settings through simple dropdowns

Download reports in multiple formats

Never need to touch code if you don't want to

## 📈 The Forecasting Approach

Exponential Smoothing (core forecasting principle)

Runs multiple models simultaneously:

Holt-Winters Exponential Smoothing (for trends & seasonality)

ARIMA time series model (classical statistical approach)

Random Forest regression (machine learning approach)

The code then combines predictions, weighting each model based on how well it fits your specific data. 

## 🔬 Bottom Line:
ForecastHub's model selection logic is:

Data-driven (not arbitrary)

Uses statistical criteria (AIC, seasonality tests)

Has proper validation (cross-validation for ML)

Follows best practices (minimum data, error handling)

Matches methodology used in published forecasting research

## 📁 Data Format Needed
Your sales data should be either a CSV or Excel file with these three columns:

## Excel

date (any date format) - product (product names as text) - sales (numbers, can have decimals)
 
## csv

date,product,sales
2023-01-31,Product A,1200.50


## 📊 What You Get
The tool generates:

CSV files with forecasts and historical data

Excel reports with multiple analysis sheets

PDF summaries for quick sharing

Charts visualizing trends

Error margins and confidence scores indicating forecast reliability

## 📁 Output Files Location
When you run forecasts, all generated files are automatically saved to a "forecast_reports" folder created by the tool.

Folder Behavior:
✅ Automatically created if it doesn't exist

📅 Files are timestamped (monthly.csv, report.xlsx, etc.)

🔄 New runs add files without deleting old ones

