This code subsumes the historical data of consumer enrollment in the insurance plans of the Affordable Care Act (ACA) at the US county level from the year 2015 through 2023 to predict the consumer enrollment in the ACA for each county for the year 2024.
The prediction is performed using several different machine learning models including Random Forest, Linear Regression, Decision Trees, Neural Network and the XGBoost model.
The machine learning models with the best performance are subsequently stacked to create an ensemble model that is also used to predict the 2024 consumer enrollment in the ACA.
The predicted values from each model and the stacked ensemble are exported in a file and then comapared with the file containing the actual consumer enrollment in the ACA in 2024 for each county to evaluate performance and accuracy.
The performance and accuracy of each model and the stacked ensemble model are evaluated using different standard measures of model performance and accuracy.
