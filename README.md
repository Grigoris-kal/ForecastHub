# ForecastHub: Local Python Sales Forecasting Tool
A Python-based sales forecasting tool that runs entirely on your computer. No cloud connectivity required, as a result, no sensitive data leaving your machine. Also included is a Graphical User Interface so that the user may upload and download one's data without touching any code and thus offering a user friendly and code-safe forecasting tool.

## 🔧 How It Works (Technically)
Built with the below Python libraries:

Pandas loads and cleans your sales data

Statsmodels handles time series forecasting (ARIMA, Exponential Smoothing)

Scikit-learn provides machine learning components (Random Forest)

Tkinter creates the graphical interface so you don't need to deal with code code. With a few clicks historical data can easily be uploaded and forecasts downloaded. 

The forecast tool analyzes historical sales, detects patterns and trends, and generates forecasts with confidence intervals.

## 🖱️ GUI Interface attached (No Coding Needed)
Included within the code is a graphical interface permititng to upload and download data easily with only a few clicks.  

Import CSV/Excel files with a few clicks

Adjust settings through simple dropdowns

Download reports in multiple formats

First launch might take a moment (Python libraries load initially)

Never need to touch code if you don't want to

## 📈 The Forecasting Approach
Runs multiple models simultaneously:

Holt-Winters Exponential Smoothing (for trends & seasonality)

ARIMA time series model (classical statistical approach)

Random Forest regression (machine learning approach)

Then it combines their predictions, weighting each model based on how well it fits your specific data. The result is an "ensemble forecast" that's typically more robust than any single model alone.

## 📁 Data Format Needed
Your sales data should be either a CSV or Excel file with these three columns:

Excel
date (any date format)

product (product names as text)

sales (numbers, can have decimals)

csv
date,product,sales
2023-01-31,Product A,1200.50
2023-01-31,Product B,850.00
2023-02-28,Product A,1350.75

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

🚀 Getting Started
Install Python (3.8 or newer)

Install required libraries: pip install pandas numpy statsmodels scikit-learn matplotlib seaborn fpdf openpyxl

Run the tool: python ForecastHub.py

Load your sales data through the GUI

Or use the console mode: python ForecastHub.py --console

📄 License
Open source - use it, modify it, share it.

