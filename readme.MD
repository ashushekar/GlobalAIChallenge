# Global AI Challenge - ESG Financial Performance

Authors: [Ashwin Naidu]() and [Siddharth Srinivasan]()

## Problem Statement
_____________________

In this project, we will be using the Global AI Challenge dataset to predict the financial performance for a given year based on ESG attributes. 
The dataset contains almost 100000 records from various shareclasses or asset managers.

Let us divide our approach into the following steps:

1. Data Analysis: We will be analyzing the data to understand the distribution of the data. We will perform some data analysis on it with respect data completeness and data uniqueness.
There are numerous ordinal, categorical, and numerical features in the dataset. Here we will try to visualize this data through some plots.

2. Feature Engineering: We will be performing some feature engineering on the dataset. We will be using the ordinal features to create new features. We will also be using the categorical features to create new features. 
We will also be using the numerical features to create new features using some log or exponential transformation methods. We will also target on categorical columns like Grades or Nominal values.

3. Model Building and Evaluation: We will be using the following models to predict the financial performance for a given year based on ESG attributes:
    - Neural Network
    - kNN
    - Ridge Regression
    - Voting Regressor

4. Prediction: We will be using the models to predict the unseen (10 %) financial performance for a given year based on ESG attributes.

## 1. Data Analysis
_____________________


```python
Notebook used: I_ESG_FP_Data_Analysis.ipynb
```

Total number of columns | Total number of rows
--- | ---
118 | 101250

### 1.1. Distribution of target variable

![image](https://user-images.githubusercontent.com/4510984/137637201-1b5b1b9f-5b1a-4b0e-8b1a-1f1b1b2b1b1b.png){:height="50%" width="50%"}
Add image 2B

### 1.2. Identify Data Completeness
Data completeness is the percentage of non-null values in the dataset. We will be using the following formula to calculate the data completeness:

```python
column_stats['data_completeness'] = df.notnull().sum(axis=0) / df.shape[0]*100
```
Add image 2C
We will remove the columns with data completeness less than 50%.

### 1.3. Identify Data Uniqueness
Data uniqueness is the percentage of unique values in the dataset. We will be using the following formula to calculate the data uniqueness:

```python
column_stats['data_uniqueness'] = df.nunique(axis=0) / df.shape[0]*100
```
Add image 2D

### 1.4. Identify Ordinal, Categorical, discrete and continuous Features
We will group all the features into the following categories:
- Ordinal Features
- Categorical Features
- Discrete Features
- Continuous Features
- Date Features

### 1.5. Check for duplicates with multi-index
We will use Shareclass name and Financial Performance as-of date as the multi-index to check for duplicates and will drop those duplicates by keeping last.

### 1.6. Removing records having partial information
We will remove the records having partial information. We will remove the records having more than 50% of the columns as null.

### 1.7. Forward fill the missing dates
It is observed that only 4 records is missing with Financial Performance as-of date. We will forward fill the missing dates from previous dates.


### 2. Feature Engineering

```python
Notebook used: II_ESG_FP_Feature_Engineering.ipynb
```

Below are the steps we will be performing for feature engineering:
- Create new features using ordinal features - Grade columns - Label encoding
- Preprocessing for numerical data 
  - Replace null values with median
  - Features with higher number of 0-values will be identified and then boolean features will be created for those features suffixed with '_isempty'
  - Using Gaussian distribution, we will try to delete the outliers
  - Using log transformation, we will try to normalize the data from continuous columns to reduce skewness
  - Using exponential transformation, we will try to normalize the data from discrete columns to reduce skewness
- Preprocessing for categorical data
  - Using one-hot encoding, we will convert the categorical data to numerical data

### 3. Model Building and Evaluation

```python
Notebook used: III_ESG_FP_Model_Selection.ipynb
```

#### 3A. Features Selection

We have exercised model training with all the features and also with the features selected using the following methods:

1. Using SelectKBest to select the top 10 features based on the k highest scores.
2. Using SelectKBest to select the top 30 features based on the k highest scores.
3. Using only grade features
4. Using all features

Top 30 features selected using SelectKBest are as follows:


We will be using the following models to predict the financial performance for a given year based on ESG attributes:
- Neural Network
- kNN
- Ridge Regression
- Voting Regressor

We will save this models and plots using mlflow.

### 4. Prediction

```python
Notebook used: IV_ESG_FP_Prediction.ipynb
```

We will test this notebook on 10 % unseen data and will predict the unseen financial performance for a given year based on ESG attributes. 
  






