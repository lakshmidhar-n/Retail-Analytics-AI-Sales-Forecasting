# Retail Analytics & AI-Powered Sales Forecasting System

## 📌 Project Overview

The **Retail Analytics & AI-Powered Sales Forecasting System** is a data analytics project designed to analyze retail sales data, identify business trends, understand store and product performance, and forecast future sales.

The project uses Python, data analysis, machine learning, and time-series forecasting techniques to transform raw retail data into meaningful insights. An interactive dashboard developed using Streamlit allows users to explore sales performance, product categories, store segments, forecasting results, and discount-related insights.

This project demonstrates how data analytics and predictive modeling can support data-driven decision-making in the retail industry.

## 🎯 Project Objectives

- Analyze retail sales performance across different stores and product categories.
- Identify sales trends, patterns, and seasonal variations.
- Understand store-wise performance and product sales patterns.
- Apply machine learning techniques to identify groups of similar stores.
- Forecast future sales using time-series forecasting methods.
- Examine discount-related patterns and their relationship with sales.
- Develop an interactive dashboard to present business insights visually.
- Demonstrate the practical application of data analytics in retail business analysis.

## 📊 Dataset Description

The project uses a retail sales dataset containing approximately **73,000 records** covering the period from **January 2023 to December 2024**.

**Dataset filename:** `Retail_Sales_Data.csv`

The dataset is used to explore sales performance, product categories, store-level patterns, and trends over time.

The quality and scope of the analysis depend on the available dataset columns, data preparation, and analytical methods used.

## ✨ Key Features

### 1. Sales Analysis

- Analyze overall retail sales performance.
- Explore sales trends over time.
- Identify changes and patterns in business performance.

### 2. Product Analysis

- Compare sales across product categories.
- Identify categories contributing to retail sales.
- Explore product mix and category-level trends.

### 3. Store Analysis

- Compare sales performance across stores.
- Apply clustering techniques to group stores based on selected characteristics.
- Visualize store segments for performance comparison.

### 4. Sales Forecasting

- Forecast future sales using time-series techniques.
- Compare forecasts with historical sales patterns.
- Analyze moving averages and weekday sales profiles.

### 5. Discount Analysis

- Explore discount-related sales patterns.
- Compare sales across available discount conditions.
- Identify patterns that may support further business analysis.

### 6. Data Integrity & Methodology

- Present information about the dataset and analytical methods.
- Explain the analytical approach used in the project.
- Review data quality and methodological limitations where applicable.

### 7. Interactive Dashboard

The Streamlit dashboard contains the following sections:

- **Briefing:** Summary of important business insights.
- **Pulse:** Sales trends and performance analysis.
- **Mix:** Product and category analysis.
- **Stores:** Store comparisons and segmentation.
- **Forecast:** Sales forecasting results.
- **Discount Lab:** Discount-related analysis.
- **Integrity & Method:** Dataset integrity and analytical methodology.

## 🛠️ Technologies Used

- **Python** – Programming and data analysis
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computations
- **Matplotlib / Plotly** – Data visualization, depending on the implementation
- **Scikit-learn** – Machine learning and clustering
- **Statsmodels** – Statistical analysis and time-series forecasting
- **Streamlit** – Interactive dashboard development
- **Jupyter Notebook / Google Colab** – Data analysis and experimentation

## 🤖 Machine Learning & Analytical Techniques

The project incorporates the following techniques:

- **K-Means Clustering:** Groups stores based on selected features.
- **Principal Component Analysis (PCA):** Reduces dimensionality to help visualize data patterns.
- **Ridge Regression:** Uses regularized regression to model relationships between variables.
- **Holt-Winters Forecasting:** Models time-series patterns using level, trend, and seasonality where applicable.
- **Moving Average Analysis:** Examines recent sales trends over a rolling time window.
- **Weekday Profile Analysis:** Compares sales patterns across different days of the week.

These methods help transform historical retail data into interpretable insights and forecast estimates.

## 📁 Project Structure

```text
retail-analytics-sales-forecasting/
│
├── app.py
├── theme.css
├── requirements.txt
├── Retail_Sales_Data.csv
├── Retail_Sales.ipynb
├── README.md
│
└── screenshots/
    ├── dashboard-overview.png
    ├── sales-analysis.png
    ├── product-analysis.png
    ├── store-analysis.png
    ├── sales-forecast.png
    ├── discount-analysis.png
    └── data-integrity.png
```

### File Descriptions

- `app.py` – Main Streamlit dashboard application.
- `theme.css` – Custom styling for the dashboard.
- `requirements.txt` – Required Python libraries.
- `Retail_Sales_Data.csv` – Retail sales dataset.
- `Retail_Sales.ipynb` – Notebook containing data analysis and modeling.
- `README.md` – Project documentation.
- `screenshots/` – Folder for dashboard screenshots.

## 📸 Dashboard Screenshots

### 1. Dashboard Overview
<img width="1902" height="872" alt="Dashboard-overview" src="https://github.com/user-attachments/assets/5ee7fcd5-2f35-46bb-a384-25050ecebcfe" />

### 2. Sales Analysis
<img width="1523" height="662" alt="Sales-analysis" src="https://github.com/user-attachments/assets/13a1f254-a650-4917-9896-f3a2f623cc58" />

### 3. Product Analysis
<img width="1248" height="806" alt="Product-analysis" src="https://github.com/user-attachments/assets/fd3b14d1-e181-4f2a-a986-c9aa8d7b207f" />

### 4. Store Analysis
<img width="1443" height="862" alt="Store-analysis" src="https://github.com/user-attachments/assets/ce432bc4-73fd-4e59-8e9d-65da961cbf7b" />

### 5. Sales Forecasting
<img width="1597" height="870" alt="Sales-forecast" src="https://github.com/user-attachments/assets/92d406c2-8ded-4c1e-a3ed-2073f5e2db1e" />

### 6. Discount Analysis
<img width="1248" height="871" alt="Discount-analysis" src="https://github.com/user-attachments/assets/c8a62498-0112-415a-8810-403e3acfca81" />

### 7. Data Integrity & Methodology
<img width="1251" height="762" alt="Data-integrity" src="https://github.com/user-attachments/assets/a1d276fd-4524-437e-be93-c678b1ad7b18" />

## ⚙️ Installation and Setup

Follow the steps below to run the project locally.

### Prerequisites

Make sure the following are installed:

- Python 3.9 or a compatible version supported by the project dependencies
- Git
- A code editor such as Visual Studio Code

### Step 1: Clone the Repository

Replace `YOUR_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_USERNAME/retail-analytics-sales-forecasting.git
```

### Step 2: Navigate to the Project Folder

```bash
cd retail-analytics-sales-forecasting
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard should open in your browser. If it does not, use the local URL displayed in the terminal, commonly:

```text
http://localhost:8501
```

## 📈 Expected Outcomes

The project aims to provide:

- Insights into overall retail sales performance.
- Better understanding of product category and store-level patterns.
- Store segmentation using machine learning.
- Future sales estimates based on historical data.
- Visual analysis of discount-related patterns.
- An interactive dashboard for exploring business performance.
- A practical demonstration of data-driven retail analysis.

Forecasts represent estimates based on historical data and the selected methods. Actual future sales may differ.

## 🎓 Learning Outcomes

This project provides practical experience in:

- Data cleaning and preprocessing.
- Exploratory Data Analysis (EDA).
- Data visualization and interpretation.
- Machine learning and clustering.
- Regression analysis.
- Time-series forecasting.
- Interactive dashboard development using Streamlit.
- Business intelligence and data-driven decision-making.

## 👩‍💻 Author

**Lakshmidhar N**

- **Project Type:** Major Project
- **Domain:** Data Analytics
- **Project Title:** Retail Analytics & AI-Powered Sales Forecasting System

## 📌 Conclusion

The Retail Analytics & AI-Powered Sales Forecasting System demonstrates the application of data analytics, machine learning, and time-series forecasting to retail sales data. By combining analytical techniques with an interactive Streamlit dashboard, the project provides a practical way to explore sales performance, identify patterns, compare store and product performance, and estimate future sales.

The project highlights the importance of transforming raw data into meaningful business insights to support informed decision-making in the retail industry.

---

*This project was developed for educational purposes to demonstrate the application of data analytics, machine learning, and interactive visualization in retail business analysis.*
