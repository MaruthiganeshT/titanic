AI & ML Internship - Task 1: Data Cleaning & Preprocessing

This repository contains the solution for Task 1 of the Elevate Labs AI & ML Internship. The task involves performing data cleaning and preprocessing on the Titanic dataset.

Objective

The goal of this task is to clean and prepare the raw Titanic dataset to make it suitable for training a machine learning model.

Tools Used

Python: Core programming language.

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Matplotlib & Seaborn: For data visualization, specifically for detecting outliers.

Scikit-learn: For feature scaling (Standardization).

Preprocessing Steps Followed

1.Data Loading and Initial Exploration:

The dataset was loaded from a URL using Pandas.

df.info(), df.head(), and df.isnull().sum() were used to get an initial overview of the data, including missing values and data types.

2.Handling Missing Values:

Age: Missing Age values were filled with the median age of the column. The median is more robust to outliers than the mean.

Cabin: The Cabin column was dropped entirely because it contained too many missing values to be useful.

Embarked: The few missing Embarked values were filled with the mode (the most frequently occurring value).

3.Encoding Categorical Features:

Sex: The Sex column was converted into numerical format using label encoding (male: 0, female: 1).

Embarked: The Embarked column was converted into numerical format using one-hot encoding to create separate binary columns for each port, avoiding any false ordinal relationship.

4.Outlier Detection and Removal:

Boxplots were generated for the Age and Fare columns to visualize outliers.

Outliers were removed using the Interquartile Range (IQR) method. Data points that fell below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR were considered outliers and removed.

5.Feature Scaling:

The non-essential Name and Ticket columns were dropped.

Numerical features (Age, Fare, Pclass, etc.) were scaled using Standardization (StandardScaler from scikit-learn). This transforms the data to have a mean of 0 and a standard deviation of 1, which is beneficial for many machine learning algorithms.

Final Result

The final output is a clean, preprocessed dataset (X) and its corresponding target variable (y), which are now ready to be used for training a predictive model.
