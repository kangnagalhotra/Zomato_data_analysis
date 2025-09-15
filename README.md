# Zomato Data Analysis

## Overview
This project analyzes the Zomato restaurant dataset to extract insights about restaurants, ratings, cuisines, and cities. It also builds a predictive model to estimate restaurant ratings based on key features.

**Tools & Libraries:** Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn  

---

## Dataset
The dataset includes information on restaurants, such as:
- Restaurant ID & Name
- City & Location
- Cuisines
- Average cost for two
- Ratings and votes
- Online ordering & table booking availability

---

## Steps Performed

### 1. Data Cleaning
- Standardized column names.
- Converted ratings to numeric values.
- Removed commas from cost fields.
- Handled missing values for cuisines and ratings.

### 2. Exploratory Data Analysis (EDA)
- Distribution of ratings and rating texts.
- Top cuisines and cities by restaurant count.
- Online ordering availability.
- Visualizations with **Seaborn** and **Matplotlib**.

### 3. Predictive Modeling
- Selected features: `average_cost_for_two`, `votes`, `city`, `cuisines`.
- Encoded categorical variables using `LabelEncoder`.
- Trained a `RandomForestRegressor` to predict restaurant ratings.
- Evaluated the model using **Mean Squared Error (MSE)** and **R² Score**.

### 4. Insights
- **Top 5 Cities by Average Rating**
- **Top 5 Cuisines by Average Rating**
- Useful for restaurant business decisions and marketing strategies.

---
## How to Run
1. Clone the repository:
git clone https://github.com/kangnagalhotra/Zomato_data_analysis.git
2. Install dependencies:
pip install -r requirements.txt
3.Run the analysis script:
python Scripts/Analysis.py

Skills Highlighted
Data Cleaning & Preprocessing
Exploratory Data Analysis
Data Visualization (Seaborn, Matplotlib)
Machine Learning (Random Forest Regression)
Feature Engineering
Insights & Reporting

