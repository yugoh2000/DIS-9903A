#!/usr/bin/env python
# coding: utf-8

# # Predicting Enrollment in the ACA

# Step-by-Step Coding Guide

# Step 1: Loading and Preparing the Dataset

# In[ ]:


# Importing the appropriate Python libraries for data loading and preparation 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Loading the datasets
historical_data = pd.read_csv("~/Documents/PhD/PhD-DS/Dissertation Files/2015-2023 ACA Enrollment.csv")
actual_data_2024 = pd.read_csv("~/Documents/PhD/PhD-DS/Dissertation Files/2024 ACA Enrollment.csv")

# Displaying the columns to identify categorical and numerical features
print(historical_data.columns)

# Double Checking for the presence of any non-numeric data
print(historical_data.dtypes)

# Displaying the first few rows of the dataset
historical_data.head()

# Removing the unnecessary columns including 'Enrolled' as predictor
X = historical_data.drop(columns=['Year', 'FIPS Code', 'County Name', 'State', 'Enrolled'])
y = historical_data['Enrolled']

# Removing rows with missing and undefined values in the target variable
X = X[y.notna()]
y = y[y.notna()]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 2: Selecting, Training and Evaluating Machine Learning Models

# Random Forest Model

# In[14]:


# Importing the appropriate Python libraries for Random Forest model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Initializing and training the Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions
rf_predictions = rf_model.predict(X_test)

# Evaluating the model
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

rf_metrics = {'MSE': rf_mse, 'MAE': rf_mae, 'R2 Score': rf_r2}
print("Random Forest Metrics:", rf_metrics)


# Linear Regression Model

# In[16]:


# Importing the appropriate Python libraries the Linear Regression model
from sklearn.linear_model import LinearRegression

# Initializing and training the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Making predictions
linear_predictions = linear_model.predict(X_test)

# Evaluating the model
linear_mse = mean_squared_error(y_test, linear_predictions)
linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_r2 = r2_score(y_test, linear_predictions)

linear_metrics = {'MSE': linear_mse, 'MAE': linear_mae, 'R2 Score': linear_r2}
print("Linear Regression Metrics:", linear_metrics)


# Decision Tree Model

# In[18]:


# Importing the appropriate Python libraries for the Decision Tree model
from sklearn.tree import DecisionTreeRegressor

# Initializing and training the Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Making predictions
dt_predictions = dt_model.predict(X_test)

# Evaluating the model
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_r2 = r2_score(y_test, dt_predictions)

dt_metrics = {'MSE': dt_mse, 'MAE': dt_mae, 'R2 Score': dt_r2}
print("Decision Tree Metrics:", dt_metrics)


# Neural Network Model

# In[20]:


# Importing the appropriate Python libraries for the Neural Network model
from sklearn.neural_network import MLPRegressor

# Initializing and training the Neural Network model
nn_model = MLPRegressor(random_state=42, max_iter=1000)
nn_model.fit(X_train, y_train)

# Making predictions
nn_predictions = nn_model.predict(X_test)

# Evaluating the model
nn_mse = mean_squared_error(y_test, nn_predictions)
nn_mae = mean_absolute_error(y_test, nn_predictions)
nn_r2 = r2_score(y_test, nn_predictions)

nn_metrics = {'MSE': nn_mse, 'MAE': nn_mae, 'R2 Score': nn_r2}
print("Neural Network Metrics:", nn_metrics)


# XGBoost Model

# In[22]:


# Importing the appropriate Python libraries for the XGBoost model
from xgboost import XGBRegressor

# Initializing and training the XGBoost model
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# Making predictions
xgb_predictions = xgb_model.predict(X_test)

# Evaluating the model
xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_r2 = r2_score(y_test, xgb_predictions)

xgb_metrics = {'MSE': xgb_mse, 'MAE': xgb_mae, 'R2 Score': xgb_r2}
print("XGBoost Metrics:", xgb_metrics)


# Step 3: Displaying the Model Evaluation Metrics

# In[24]:


# Displaying the Model Evaluation Metrics
models_metrics = {
    'Random Forest': rf_metrics,
    'Linear Regression': linear_metrics,
    'Decision Tree': dt_metrics,
    'Neural Network': nn_metrics,
    'XGBoost': xgb_metrics
}
metrics_df = pd.DataFrame(models_metrics).T
print(metrics_df)


# Step 4: Feature Engineering with SHAP Analysis and Plot

# In[27]:


# Importing the appropriate Python libraries for SHAP
import shap

# SHAP Analysis for the best model (e.g., Linear Regression)
explainer = shap.Explainer(linear_model, X_train)
shap_values = explainer(X_test)

# Plot SHAP values
shap.summary_plot(shap_values, X_test)


# Step 5: Creating the Stacked Ensemble Model

# In[29]:


# Importing the appropriate Python libraries for creating the Stacked Ensemble model
from sklearn.ensemble import StackingRegressor

# Creating a stacked model using the best models (e.g., Linear Regression and Random Forest)
estimators = [
    ('rf', rf_model),
    ('linear', linear_model)
]

stacked_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stacked_model.fit(X_train, y_train)

# Making predictions
stacked_predictions = stacked_model.predict(X_test)

# Evaluating the model
stacked_mse = mean_squared_error(y_test, stacked_predictions)
stacked_mae = mean_absolute_error(y_test, stacked_predictions)
stacked_r2 = r2_score(y_test, stacked_predictions)

stacked_metrics = {'MSE': stacked_mse, 'MAE': stacked_mae, 'R2 Score': stacked_r2}
print("Stacked Model Metrics:", stacked_metrics)


# Step 6: Creating a visualization of the Stacked Ensemble Model

# In[31]:


# Importing the appropriate Python libraries for analysis and visualizations
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# Generating a small subset of the data for quick visualization
X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Fitting the stacked model on the smaller subset
stacked_model.fit(X_train_small, y_train_small)
stacked_predictions_small = stacked_model.predict(X_test_small)

# Evaluating the model on the smaller subset
stacked_mse_small = mean_squared_error(y_test_small, stacked_predictions_small)

# Visualizing the model predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_small, stacked_predictions_small, color='blue', label='Predictions')
plt.plot([y_test_small.min(), y_test_small.max()], [y_test_small.min(), y_test_small.max()], 'k--', lw=2, label='Ideal')
plt.xlabel('Actual Enrollment')
plt.ylabel('Predicted Enrollment')
plt.title(f'Stacked Regressor Predictions vs Actuals (MSE: {stacked_mse_small:.2f})')
plt.legend()
plt.grid(True)
plt.show()


# Creating a visualization of the Stacked Regressor

# In[33]:


# Importing the appropriate Python libraries for analysis and visualizations
from sklearn.ensemble import StackingRegressor
import matplotlib.pyplot as plt
from sklearn import set_config
import graphviz
from sklearn.tree import export_graphviz
from IPython.display import display

# Creating a stacked model using the best models
estimators = [
    ('xgboost', XGBRegressor(random_state=42)),
    ('random_forest', RandomForestRegressor(random_state=42)),
    ('linear_regression', LinearRegression()),
    ('decision_tree', DecisionTreeRegressor(random_state=42)),
    ('neural_network', MLPRegressor(random_state=42, max_iter=1000))
]

stacked_model = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(random_state=42))
stacked_model.fit(X_train, y_train)

# Making predictions
stacked_predictions = stacked_model.predict(X_test)

# Evaluating the model
stacked_mse = mean_squared_error(y_test, stacked_predictions)
stacked_mae = mean_absolute_error(y_test, stacked_predictions)
stacked_r2 = r2_score(y_test, stacked_predictions)

stacked_metrics = {'MSE': stacked_mse, 'MAE': stacked_mae, 'R2 Score': stacked_r2}
print("Stacked Model Metrics:", stacked_metrics)

# Visualizing the structure of the stacking model
set_config(display='diagram')
display(stacked_model)


# Step 7: Perfoming Hyperparameter Tuning with Optuna

# In[35]:


# Importing the appropriate Python libraries for Optuna
import optuna

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 1, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
    }
    
    model = XGBRegressor(**param)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Displaying the trial results of the Optuna experiment
optuna_results = {
    'Best Trial': study.best_trial,
    'Number of Trials': len(study.trials)
}
print("Optuna Results:", optuna_results)


# Step 8: Evaluating the Model with Additional Metrics

# In[37]:


# Importing the appropriate Python libraries for the additional evaluation metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def calculate_metrics(y_true, y_pred):
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2 Score': r2_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true.round(), y_pred.round()),
        'Precision': precision_score(y_true.round(), y_pred.round(), average='macro'),
        'Recall': recall_score(y_true.round(), y_pred.round(), average='macro'),
        'F1 Score': f1_score(y_true.round(), y_pred.round(), average='macro')
    }

# Evaluating the models using the additional metrics
models = {
    'Random Forest': rf_model,
    'Linear Regression': linear_model,
    'Decision Tree': dt_model,
    'Neural Network': nn_model,
    'XGBoost': xgb_model,
    'Stacked Model': stacked_model
}

additional_metrics = {name: calculate_metrics(y_test, model.predict(X_test)) for name, model in models.items()}
additional_metrics_df = pd.DataFrame(additional_metrics).T
print(additional_metrics_df)


# In[38]:


Step 9: Plotting the Threshold and F1-Score Chart


# In[39]:


# Importing the appropriate Python libraries for analysis and visualizations
import matplotlib.pyplot as plt

thresholds = np.arange(0.1, 1.0, 0.1)
f1_scores = []

for thresh in thresholds:
    y_pred_thresh = (stacked_predictions >= thresh).astype(int)
    f1_scores.append(f1_score(y_test.round(), y_pred_thresh, average='macro'))

plt.plot(thresholds, f1_scores, marker='o')
plt.title('Threshold vs F1-Score')
plt.xlabel('Threshold')
plt.ylabel('F1-Score')
plt.grid(True)
plt.show()


# Plotting the ROC Curve and AUC Scores

# In[41]:


# Importing the appropriate Python libraries the visualizations
from sklearn.metrics import roc_curve, roc_auc_score

# Setting a threshold to binarize the target values
threshold = y_test.mean()

# Binarizing the target values
y_test_binarized = (y_test >= threshold).astype(int)

# Computing ROC curve and ROC area for each model
plt.figure(figsize=(10, 8))

for name, model in models.items():
    y_pred_proba = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test_binarized, y_pred_proba)
    roc_auc = roc_auc_score(y_test_binarized, y_pred_proba)
    
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# Step 10: Predicting 2024 Enrollment Values and Comparing them with the Actual Enrollment Data

# In[44]:


# Predicting the 2024 enrollment using the trained models
X_2024 = actual_data_2024.drop(columns=['Year', 'FIPS Code', 'County Name', 'State', 'Enrolled'])

# Creating a DataFrame to hold the predictions from each model
predictions_df = pd.DataFrame({
    'FIPS Code': actual_data_2024['FIPS Code'],
    'Actual Enrollment 2024': actual_data_2024['Enrolled'],
    'Random Forest': rf_model.predict(X_2024),
    'Linear Regression': linear_model.predict(X_2024),
    'Decision Tree': dt_model.predict(X_2024),
    'Neural Network': nn_model.predict(X_2024),
    'XGBoost': xgb_model.predict(X_2024),
    'Stacked Model': stacked_model.predict(X_2024)
})

# Calculating the percentage accuracy for each model
def calculate_accuracy(actual, predicted):
    return 100 * (1 - np.abs((actual - predicted) / actual))

predictions_df['RF Accuracy (%)'] = calculate_accuracy(predictions_df['Actual Enrollment 2024'], predictions_df['Random Forest'])
predictions_df['LR Accuracy (%)'] = calculate_accuracy(predictions_df['Actual Enrollment 2024'], predictions_df['Linear Regression'])
predictions_df['DT Accuracy (%)'] = calculate_accuracy(predictions_df['Actual Enrollment 2024'], predictions_df['Decision Tree'])
predictions_df['NN Accuracy (%)'] = calculate_accuracy(predictions_df['Actual Enrollment 2024'], predictions_df['Neural Network'])
predictions_df['XGB Accuracy (%)'] = calculate_accuracy(predictions_df['Actual Enrollment 2024'], predictions_df['XGBoost'])
predictions_df['Stacked Accuracy (%)'] = calculate_accuracy(predictions_df['Actual Enrollment 2024'], predictions_df['Stacked Model'])

# Displaying the first few rows of the DataFrame
print(predictions_df.head())

# Defining and Saving the predictions to a CSV file
predictions_df.to_csv('~/Documents/PhD/PhD-DS/Dissertation Files/Predicted_Enrollment_2024_with_Accuracy_Using_2015-2023_Data.csv', index=False)


# Visualizing the Accuracy of Predictions

# In[46]:


# Importing the appropriate Python libraries for anaysis and visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Creating a DataFrame to hold the accuracy scores
accuracy_df = predictions_df[['FIPS Code', 'RF Accuracy (%)', 'LR Accuracy (%)', 'DT Accuracy (%)', 'NN Accuracy (%)', 'XGB Accuracy (%)', 'Stacked Accuracy (%)']]

# Melting the DataFrame for easier plotting
accuracy_melted_df = accuracy_df.melt(id_vars='FIPS Code', var_name='Model', value_name='Accuracy')

# Plotting the accuracies
plt.figure(figsize=(12, 8))
sns.boxplot(x='Model', y='Accuracy', data=accuracy_melted_df)
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[ ]:




