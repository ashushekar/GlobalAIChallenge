"""
Dataset include the following details:
Fund information: general information about the fund
Financial results: our target variable is contained in that group Financial performance: Financial performance as of date
    Financial performance: Month end trailing returns, year 1, year 3, year 5, year 10 [target variable]
ESG grading metrics: overall grading by the FFF organization
    Fossil Free Funds
    Deforestation Free Funds
    Gender Equality Funds
    Gun Free Funds
    Weapons Free Funds
    Tobacco Free Funds
    Prison Free Funds
Detailed breakdown of the ESG gr    ading metrics
"""
import mlflow
import numpy as np
import pandas as pd
# Import visualization libraries
import plotly.graph_objects as go

# settings to display in pycharm


# import HTML for pretty printing

# Read in the Shareclasses excel data with openpyxl
shareclasses_df = pd.read_excel('data/fossilfund_dataset.xlsx', engine='openpyxl', sheet_name='Shareclasses')

# Now read the key data from the excel file
key_df = pd.read_excel('data/fossilfund_dataset.xlsx', engine='openpyxl', sheet_name='Key', header=None,
                       names=['category', 'feature', 'feature description'])

# **********************************************************************************************************************
# 1. Let us print all related info about the shareclasses
# **********************************************************************************************************************
print('*******************************************************************************************************')
print("Total number of columns in the shareclasses_df: ", len(shareclasses_df.columns))
print("No. of columns containing null values", len(shareclasses_df.columns[shareclasses_df.isna().any()]))

print("No. of columns not containing null values", len(shareclasses_df.columns[shareclasses_df.notna().all()]))
print("No. of numerical columns ", len(shareclasses_df.select_dtypes(np.number).columns))

print("Total no. of rows in the dataframe", len(shareclasses_df))
print('*******************************************************************************************************')


# **********************************************************************************************************************
# 2. Let us convert all date columns to datetime format
# **********************************************************************************************************************
def convert_to_datetime(inDF):
    """
    This function converts all date columns to datetime format
    """
    outDF = inDF.copy()
    all_date_cols = []
    for col in outDF.columns:
        if ' date' in col.lower():
            outDF[col] = pd.to_datetime(outDF[col], format='%Y-%m-%d', errors='coerce')
            all_date_cols.append(col)
    return outDF, all_date_cols


shareclasses_df, all_date_cols = convert_to_datetime(shareclasses_df)
print("All date columns in the shareclasses_df: ", shareclasses_df[all_date_cols].sample(5))

# **********************************************************************************************************************
# 3. Let us check data completeness
# **********************************************************************************************************************
print('*******************************************************************************************************')
columns_info = pd.DataFrame()
# Let us calculate fill rate for each column rowise and sort it by fill rate
columns_info['fill_rate'] = shareclasses_df.notnull().sum(axis=0) / len(shareclasses_df) * 100
columns_info = columns_info.sort_values(by='fill_rate')

# print columns with fill rate less than 50%
print("Columns with fill rate less than 50%")
print(columns_info[columns_info['fill_rate'] < 50])

# now drop columns with fill rate less than 50%
new_shareclasses_df = shareclasses_df.drop(columns=columns_info[columns_info['fill_rate'] < 50].index)
print('*******************************************************************************************************')

# **********************************************************************************************************************
# 4. Let us analyse Target variable
# **********************************************************************************************************************
print("Target variables can be: ", columns_info['fill_rate'].filter(regex='Financial performance.*'))
# we will drop year 3 and year 5 as they have less than influence
new_shareclasses_df = new_shareclasses_df.drop(columns=['Financial performance: Month end trailing returns, year 3',
                                                        'Financial performance: Month end trailing returns, year 5',
                                                        'Financial performance: Month end trailing returns, year 10'])

# **********************************************************************************************************************
# 5. Let us analyse the Shareclass Inception Date
# **********************************************************************************************************************
perf_date = pd.DataFrame()
filter_df = new_shareclasses_df['Fund profile: Shareclass inception date'][(
        (new_shareclasses_df['Financial performance: Financial performance as-of date'] -\
         pd.DateOffset(years=1)) > new_shareclasses_df[
            'Fund profile: Shareclass inception date'])]
perf_date["year1"] = filter_df.groupby(filter_df.dt.year).count()
print(perf_date.tail())

# Normalize row perf_date by year
perf_date = perf_date.div(perf_date.sum(axis=1), axis=0)
# Now plot the graph
fig = go.Figure(data=[go.Bar(x=perf_date.index, y=perf_date['year1'])])
fig.update_layout(title_text='Shareclass Inception Date vs Financial performance: Financial performance as-of date')
fig.show()

# Plot correlation heatmap with seaborn and plotly

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

corr = new_shareclasses_df.corr()
# Plot correlation heatmap with plotly with figsize of 1000x1000
fig = px.imshow(corr, labels=dict(x="Columns", y="Columns", color="Correlation"))
fig.update_layout(width=1000, height=1000)
fig.show()

# Now let us find positve and negative correlation with target variable
target_var = 'Financial performance: Month end trailing returns, year 1'
corr_target = abs(corr[target_var])
# Selecting highly correlated features and sort it by correlation
relevant_features = corr_target[corr_target > 0.5].sort_values(ascending=False)

# now selecting negative correlated features
neg_corr_target = corr_target[corr_target < -0.5]
print("Highly correlated features with target variable: ", relevant_features)
print("Negative correlated features with target variable: ", neg_corr_target)

# now let us plot the heatmap of highly correlated features
# Selecting highly correlated features
corr_features = new_shareclasses_df[relevant_features.index].corr()
# Plot correlation heatmap with plotly with figsize of 1000x1000
# show legend
fig = px.imshow(corr_features, labels=dict(x="Columns", y="Columns", color="Correlation"),
                color_continuous_scale=px.colors.sequential.Plasma,
                title="Correlation heatmap of highly correlated features",
                width=1000, height=1000,
                color_continuous_midpoint=0,
                showlegend=True)

# **********************************************************************************************************************
# Now let us plot trends in target variable
# **********************************************************************************************************************
# Convert all date columns to datetime format
new_shareclasses_df, all_date_cols = convert_to_datetime(new_shareclasses_df)
# Plot trends in target variable
fig = px.line(new_shareclasses_df, x='Fund profile: Shareclass inception date',
                y='Financial performance: Month end trailing returns, year 1',
                title='Trends in target variable')
fig.show()


fig.update_layout(width=1000, height=1000)
fig.show()


new_shareclasses_df[all_date_cols] = new_shareclasses_df[all_date_cols].astype(float)

# **********************************************************************************************************************
# Split the data into train and test
# **********************************************************************************************************************
from sklearn.model_selection import train_test_split

# Select X and y variables
X = new_shareclasses_df.drop(columns=['Financial performance: Month end trailing returns, year 1'])
y = new_shareclasses_df['Financial performance: Month end trailing returns, year 1']

# Split the data into train and test
X = new_shareclasses_df.drop(columns=[target_var])
y = new_shareclasses_df[target_var]

# Now use SelectKBest to select top 30 features and iterate over different values of k
from sklearn.feature_selection import SelectKBest, f_regression

# Label encode date columns
from sklearn.preprocessing import LabelEncoder

# Label encode date columns
for col in all_date_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

#
# Create a function to select top k features
def select_features(X_train, y_train, X_test, k):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k=k)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

# Select only grade columns
grade_cols = X.filter(regex='Grade.*').columns
X_grade = X[grade_cols]

# **********************************************************************************************************************
# Now let us split into train and test with scalar approach
# **********************************************************************************************************************
def split_train_test(X, y, test_size=0.3, random_state=42):
    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Now apply scalar
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# **********************************************************************************************************************
# Now define a regression function
# **********************************************************************************************************************
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# import Ridge and Lasso
from sklearn.linear_model import Ridge, Lasso
# import gridsearchcv
from sklearn.model_selection import GridSearchCV#
# import Kfold
from sklearn.model_selection import KFold
# import r2 score
from sklearn.metrics import r2_score
# import mean squared error
from sklearn.metrics import mean_squared_error

def regression_model(X_train, y_train, X_test, y_test):
    # fit the ridge model with grid search and get RMSE, R2 score, MAE and MSE
    # define model
    model = Ridge()
    # define model evaluation method
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    # define grid
    grid = dict()
    grid['alpha'] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    # define search with scoring as Rsquared
    search = GridSearchCV(model, grid, scoring='r2', cv=cv, n_jobs=-1)

    # perform the search
    results = search.fit(X_train, y_train)
    # summarize
    print('Best Score: %s' % results.best_score_)
    print('Best Hyperparameters: %s' % results.best_params_)
    # get the best model
    model = results.best_estimator_
    # evaluate the model
    y_pred = model.predict(X_test)
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE: %.3f' % rmse)
    # calculate R2 score
    r2 = r2_score(y_test, y_pred)
    print('R2: %.3f' % r2)
    # calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    print('MAE: %.3f' % mae)
    # calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    print('MSE: %.3f' % mse)

    # Convert best model to h5 file
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    # plot the predicted values vs actual values
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title('Actual vs Predicted values')
    plt.show()

    return model

# import kNN regressor
from sklearn.neighbors import KNeighborsRegressor

def knn_model(X_train, y_train, X_test, y_test):
    # deine kNN model
    knn = KNeighborsRegressor()
    # define grid
    grid = dict()
    grid['n_neighbors'] = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    grid['weights'] = ['uniform', 'distance']
    grid['metric'] = ['euclidean', 'manhattan']
    grid['p'] = [1, 2]
    # define search with scoring as Rsquared
    search = GridSearchCV(knn, grid, scoring='r2', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(X_train, y_train)
    # summarize
    print('Best Score: %s' % results.best_score_)
    print('Best Hyperparameters: %s' % results.best_params_)
    # get the best model
    model = results.best_estimator_
    # evaluate the model
    y_pred = model.predict(X_test)
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE: %.3f' % rmse)
    # calculate R2 score
    r2 = r2_score(y_test, y_pred)
    print('R2: %.3f' % r2)
    # calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    print('MAE: %.3f' % mae)
    # calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    print('MSE: %.3f' % mse)

    return model

# Let us try with tensorflow and keras neural network approach for our regression problem
# **********************************************************************************************************************
# Now define a regression function
# **********************************************************************************************************************

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

def neural_network_model(X_train, y_train, X_test, y_test):
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=13, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    # compile the keras model with r2 score as metric
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error',
                                                                        'mean_absolute_percentage_error',
                                                                        'cosine_proximity'])

    # fit the keras model on the dataset
    model.fit(X_train, y_train, epochs=150, batch_size=10)

    # save the best model with lowest loss and checkpointing and earlystopping

    # define the checkpoint
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

    # define the early stopping
    early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='min')

    callbacks_list = [checkpoint, early_stopping]
    # fit the model with history
    history = model.fit(X_train, y_train, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)
    # evaluate the keras model
    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy))
    # make predictions
    y_pred = model.predict(X_test)
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE: %.3f' % rmse)

    # calculate R2 score
    r2 = r2_score(y_test, y_pred)
    print('R2: %.3f' % r2)
    # calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    print('MAE: %.3f' % mae)
    # calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    print('MSE: %.3f' % mse)

    # other metrics to check performance of the model
    # calculate MAPE
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print('MAPE: %.3f' % mape)

    # calculate cosine proximity
    cosine_proximity = cosine_proximity(y_test, y_pred)
    print('Cosine Proximity: %.3f' % cosine_proximity)

    # save model architecture to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # load model architecture from JSON
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
     # import model_from_json
    from tensorflow.keras.models import model_from_json
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("weights.best.hdf5")
    print("Loaded model from disk")


    # serialize weights to HDF5
    model.save_weights("model.h5")

    # plot the predicted values vs actual values
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title('Actual vs Predicted values')
    plt.show()

    plt.savefig('Actual vs Predicted values.png')
    # to mlflow
    mlflow.log_artifact("plot.png")

    # plot training history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicted vs Actual'))
    fig.update_layout(title='Predicted vs Actual values', xaxis_title='Actual values', yaxis_title='Predicted values')
    fig.show()

    # save this plotly as jpeg
    fig.write_image("Predicted vs Actual values.jpeg")


    # log all the metrics to mlflow
    mlflow.log_metrics({"RMSE": rmse, "R2": r2, "MAE": mae, "MSE": mse})

    # log model name to mlflow
    mlflow.log_param("model_name", "neural_network_model")

    # log hyperparameter to mlflow
    mlflow.log_param("epochs", 150)
    mlflow.log_param("batch_size", 10)

    return model

# call the model
model = neural_network_model(X_train, y_train, X_test, y_test)

#stop mlflow
mlflow.end_run()

# log a dictionary of metrics to mlflow
# convert dict to json for logging
metrics = {"RMSE": rmse, "R2": r2, "MAE": mae, "MSE": mse}
metrics_json = json.dumps(metrics)
mlflow.log_metric("metrics", metrics_json)

# save top 30 features to mlflow and json
top_30_features = pd.DataFrame(data=feature_importance, columns=['feature', 'importance'])
top_30_features = top_30_features.sort_values(by='importance', ascending=False).head(30)
top_30_features.to_json('top_30_features.json')

# load json to pandas
top_30_features = pd.read_json('top_30_features.json')

# load model weights
model.load_weights("weights.best.hdf5")











