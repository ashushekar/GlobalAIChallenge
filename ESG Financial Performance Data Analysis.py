"""
In this program we will load the data from the csv file and perform some analysis on it.
Their are numerous categorica, and numerical data in the csv file. We will perform some
analysis on the data and plot some graphs to visualize the data with respect to target variable.

For easy approach and retrieval we store datatypes of each columns in a json file.

Below are steps which will be performed in this program:
1. Load the data from the csv file
2. Perform some analysis on the data
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import plotly.express as px

# 1. Load the data from the csv file
dataset = pd.read_csv('ESG Financial Performance Data.csv')

# 2. Perform some analysis on the data
# 2.1. Plot the distribution of target variable with standard deviation and mean using plotly and display mean and std in the graph

fig = px.histogram(dataset, x="Financial Performance", nbins=100, title="Distribution of Financial Performance")
fig.add_vline(x=dataset["Financial Performance"].mean(), line_width=3, line_dash="dash", line_color="green",)#
fig.add_vline(x=dataset["Financial Performance"].mean() + dataset["Financial Performance"].std(), line_width=3, line_dash="dash", line_color="red")
fig.add_vline(x=dataset["Financial Performance"].mean() - dataset["Financial Performance"].std(), line_width=3, line_dash="dash", line_color="red")
fig.show()

# calculate the mean and standard deviation of the target variable
mean = dataset["Financial Performance"].mean()
std = dataset["Financial Performance"].std()

# It can be seen from figure that dataset is normally distributed around 0.5

# 3. Extract all date columns from the dataset and convert them to datetime format
date_columns = []
for column in dataset.columns:
    if dataset[column].dtype == 'object':
        date_columns.append(column)

for column in date_columns:
    dataset[column] = pd.to_datetime(dataset[column])


# Data COmpleteness Analysis
# What is data completeness?
# Data completeness is the percentage of non-missing values in a dataset.
# It is a measure of how much data is missing from a dataset. For example, if a dataset has 100 rows and 10 columns,
# and 5 rows in the 3rd column are missing, then the data completeness of that column is 95%.

column_stats = pd.DataFrame()
column_stats['fill_percent'] = dataset.count() / len(dataset) * 100
# sort it
column_stats.sort_values(by='fill_percent', ascending=False, inplace=True)
# plot it with plotly bar
fig = px.bar(column_stats, x=column_stats.index, y='fill_percent', title='Data Completeness')
fig.show()

# Remove columns with less than 50% data completeness
dataset = dataset[column_stats[column_stats['fill_percent'] > 50].index]

# Data Uniqueness Analysis
# What is data uniqueness?
# Data uniqueness is the percentage of unique values in a dataset.
# It is a measure of how many unique values are present in a dataset.
# For example, if a dataset has 100 rows and 10 columns,
# and 5 rows in the 3rd column are unique, then the data uniqueness of that column is 5%.

column_stats['unique_percent'] = dataset.nunique() / len(dataset) * 100
# sort it
column_stats.sort_values(by='unique_percent', ascending=False, inplace=True)
# plot it with plotly bar
fig = px.bar(column_stats, x=column_stats.index, y='unique_percent', title='Data Uniqueness')
fig.show()


# Now uidentify the columns which are categorical, discrete, ordinal, continuous and date
# We will store the datatypes of each columns in a json file
# We will use this json file to perform analysis on the data

# Identify the columns which are categorical
categorical_columns = []
for column in dataset.columns:
    if dataset[column].dtype == 'object':
        categorical_columns.append(column)

# Identify the columns which are discrete
discrete_columns = []
for column in dataset.columns:
    if dataset[column].dtype == 'int64':
        discrete_columns.append(column)

# Identify the columns which are ordinal
ordinal_columns = []
for column in dataset.columns:
    if dataset[column].dtype == 'int64':
        ordinal_columns.append(column)

# Identify the columns which are continuous
continuous_columns = []
for column in dataset.columns:
    if dataset[column].dtype == 'float64':
        continuous_columns.append(column)

# Identify the columns which are date
date_columns = []
for column in dataset.columns:
    if dataset[column].dtype == 'datetime64[ns]':
        date_columns.append(column)

# Store the datatypes of each columns in a json file
data_types = {
    'categorical_columns': categorical_columns,
    'discrete_columns': discrete_columns,
    'ordinal_columns': ordinal_columns,
    'continuous_columns': continuous_columns,
    'date_columns': date_columns
}

with open('data_types.json', 'w') as outfile:
    json.dump(data_types, outfile)

# Identify zolumns with percentage of 0 values
column_stats['zero_percent'] = (dataset == 0).sum() / len(dataset) * 100
# sort it
column_stats.sort_values(by='zero_percent', ascending=False, inplace=True)
# plot it with plotly bar
fig = px.bar(column_stats, x=column_stats.index, y='zero_percent', title='Percentage of 0 values')
fig.show()

# CHeck with duplicate rows
dataset.duplicated().sum()

# so no duplicate rows, but can we use multi-indexing to remove duplicate rows
# but let us identify participants for multi-indexing
dataset[ordinal_columns].nunique()

# so we can use multi-indexing on the following columns
# ['Company Name', 'Ticker', 'Year', 'Quarter']

# remove duplicate rows
dataset.drop_duplicates(subset=['Company Name', 'Ticker', 'Year', 'Quarter'], inplace=True)

# Missing Value Analysis
# We will track missing values on the basis of asset manager column. This is because grouping by as-of date will give us
# a better idea of missing values in each asset manager and we will go for stacked bar chart to visualize it

grouped_Df = dataset.groupby(['As-of Date'])
# Sort the grouped dataframe by asset manager in descending order
grouped_Df = grouped_Df['Asset Manager'].value_counts().sort_values(ascending=False).reset_index(name='count')

# now plot the count of each asset manager by plotly stacked bar chart
fig = px.bar(grouped_Df['Asset Manager'].value_counts().unstack(), title='Missing Values by Asset Manager')

# set xaixs title
fig.update_xaxes(title_text='Asset Manager')

# move legend to bottom as horizontal
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig.show()

# Replacing missing dates with the previous date through forward fill
# but first let us identify count of dates where there is no data
dataset['As-of Date'].value_counts()

# so we can see that there are 3 dates where there is no data
# so we will replace these dates with the previous date through forward fill
dataset['As-of Date'].fillna(method='ffill', inplace=True)

# we will use mlflow to store histogram plots for all the columns

# import mlflow
import mlflow

# initialize mlflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('Data Analysis')

# create a function to plot histogram for each category of columns as facet with plotly

# import make_subplots
from plotly.subplots import make_subplots
def plot_histogram(column, title):
    # identify the number of rows and columns
    rows = len(column) // 3 + 1
    columns = 3

    # create a figure of make_subplots subplot title font size of 8
    fig = make_subplots(rows=rows, cols=columns, subplot_titles=column, horizontal_spacing=0.1, vertical_spacing=0.1,
                        specs=[[{'type': 'histogram'}] * columns] * rows,
                        subplot_titles_font_size=8)
    fig = make_subplots(rows=rows, cols=columns, subplot_titles=column, horizontal_spacing=0.1, vertical_spacing=0.1,
                        specs=[[{'type': 'histogram'}] * columns] * rows, print_grid=False)

    fig = make_subplots(rows=rows, cols=columns, subplot_titles=column, horizontal_spacing=0.1, vertical_spacing=0.1,
    fig = make_subplots(rows=rows, cols=columns, subplot_titles=column)

    # add trace for each category with row and column in subplot
    for i in range(len(column)):
        row = i // 3 + 1
        column = i % 3 + 1
        fig.add_trace(go.Histogram(x=dataset[column], name=column), row=row, col=column)
        # remove legends from all subplots and update font size of title
        fig.update_layout(showlegend=False, title_font_size=20)
        fig.update_layout(showlegend=False, font=dict(size=10))
        fig.update_layout(showlegend=False)

    # update layout
    fig.update_layout(title_text=title)
    # reduce font size for all subplots

    fig.update_layout(font=dict(size=8))

    # save the figure as html
    fig.write_html(title + '.html')

    # now log the figure in mlflow
    mlflow.log_artifact(title + '.html')

# log transform using log1p the continuous columns
for column in continuous_columns:
    dataset[column] = np.log1p(dataset[column])

# plot histogram for continuous columns with before and after log transform in same plotly figure
# create a figure of make_subplots subplot title font size of 8 and calculate vertical spacing
vertical_spacing = 0.1 / (len(continuous_columns) // 3 + 1)
horizontal_spacing = 0.1 / 3

fig = make_subplots(rows=2, cols=1, subplot_titles=['Before Log Transform', 'After Log Transform'],
                    horizontal_spacing=0.1, vertical_spacing=0.1,
                    specs=[[{'type': 'histogram'}], [{'type': 'histogram'}]],
                    subplot_titles_font_size=8)

# add trace for each category with row and column in subplot
for i in range(len(continuous_columns)):
    fig.add_trace(go.Histogram(x=dataset[continuous_columns[i]], name=continuous_columns[i]), row=1, col=1)
    fig.add_trace(go.Histogram(x=dataset[continuous_columns[i]], name=continuous_columns[i]), row=2, col=1)

# remove legends from all subplots and update font size of title
fig.update_layout(showlegend=False, title_font_size=20)
fig.show()

# using mlflow sklearn to select K best features
# import mlflow sklearn
import mlflow.sklearn

# import SelectKBest
from sklearn.feature_selection import SelectKBest

# import f_regression
from sklearn.feature_selection import f_regression

# create a function to select K best features
def select_k_best_features(X, y, k):
    # create a SelectKBest object with f_regression and use mlflow
    with mlflow.start_run(run_name='Select K Best Features'):
        # create a SelectKBest object with f_regression
        select_k_best = SelectKBest(f_regression, k=k)

        # fit the select_k_best on X and y
        select_k_best.fit(X, y)

        # get the support of select_k_best
        support = select_k_best.get_support()

        # get the scores of select_k_best
        scores = select_k_best.scores_

        # create a dataframe with scores and columns
        scores_df = pd.DataFrame({'columns': X.columns, 'scores': scores})

        # sort the dataframe by scores in descending order
        scores_df = scores_df.sort_values(by='scores', ascending=False)

        # log the scores_df in mlflow
        mlflow.log_artifact(scores_df)

        # log the k in mlflow
        mlflow.log_param('k', k)

        # return the support
        return support

# we will use voting regressor to predict the target variable
# import VotingRegressor
from sklearn.ensemble import VotingRegressor

# import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# import XGBRegressor
from xgboost import XGBRegressor

# import kNN Regressor
from sklearn.neighbors import KNeighborsRegressor

# call voting regressor with RandomForestRegressor, GradientBoostingRegressor, XGBRegressor and kNN Regressor
# and select grid search cv to find the best parameters

# import GridSearchCV
from sklearn.model_selection import GridSearchCV

# create a function to call voting regressor with RandomForestRegressor, GradientBoostingRegressor, XGBRegressor and kNN Regressor
# and select grid search cv to find the best parameters

def voting_regressor(X, y):
    # create a VotingRegressor object with RandomForestRegressor, GradientBoostingRegressor, XGBRegressor and kNN Regressor
    # and select grid search cv to find the best parameters
    with mlflow.start_run(run_name='Voting Regressor'):
        # create a VotingRegressor object with RandomForestRegressor, GradientBoostingRegressor, XGBRegressor and kNN Regressor
        voting_regressor = VotingRegressor([('rf', RandomForestRegressor()), ('gb', GradientBoostingRegressor()),
                                            ('xgb', XGBRegressor()), ('knn', KNeighborsRegressor())])

        # create a dictionary of parameters for RandomForestRegressor, GradientBoostingRegressor, XGBRegressor and kNN Regressor
        params = {'rf__n_estimators': [100, 200, 300], 'rf__max_depth': [5, 10, 15],
                  'gb__n_estimators': [100, 200, 300], 'gb__max_depth': [5, 10, 15],
                  'xgb__n_estimators': [100, 200, 300], 'xgb__max_depth': [5, 10, 15],
                  'knn__n_neighbors': [5, 10, 15]}

        # create a GridSearchCV object with voting_regressor and params and scoring as r2

        grid_search_cv = GridSearchCV(voting_regressor, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

        # fit the grid_search_cv on X and y
        grid_search_cv.fit(X, y)

        # get the best estimator
        best_estimator = grid_search_cv.best_estimator_

        # get the best score
        best_score = grid_search_cv.best_score_

        # log the best score in mlflow
        mlflow.log_metric('best_score', best_score)

        # log the best estimator in mlflow
        mlflow.sklearn.log_model(best_estimator, 'best_estimator')

        # save the best estimator in a h5 file
        joblib.dump(best_estimator, 'best_estimator.h5')

        # return the best estimator
        return best_estimator

# Load the model
model = joblib.load('best_estimator.h5')