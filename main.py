#import pandas as pd
from pandas.plotting import scatter_matrix
#import seaborn as sb
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
#from sklearn import metrics, model_selection
from data_handler import data_handler
from modeler import modeler
from predictor import predictor
from visualizer import visualizer

# Instantiate the class objects
dh = data_handler()
md = modeler()
pred = predictor()
vis = visualizer()

# Load the data and send it through the washing machine
df = dh.load_data()
df = dh.clean_data(df)
df = dh.trim_demand(df, 5)

#dh.export_top(dh.item_selection_helper(df))

# Featurize the data
features = dh.data_wrangler(df)
print(features)

# ADDED FOR MULTI-MODEL SUPPORT
alt_df = df.copy()
item_dict = {item_id: item_df for item_id, item_df in alt_df.groupby('Item')}

# Split it up
X_train, X_test, y_train, y_test = md.split_data(df, features)

# Create a prediction demand week in future of training data
print(X_train['time_index'].max())

# Train a model for both Original Demand
model = md.train_model(X_train, y_train)

# TRAIN ALL THE MODELS
# all_models = {}
# for item_id, item_data in item_dict.items():
#     print(f"Training Item {item_id} model")
#     model = md.train_model(item_data)
#     all_models[item_id] = model

# Make predictions with the models
y_pred = pred.crystal_ball(model, X_test)

dh.export_top(X_test)

# Prep the results
results = dh.prep_results(X_test, y_test, y_pred)

# Get the MAPE
results['MAPE'] = np.abs((results['Actual Demand'] - results['Predicted Demand']) / results['Actual Demand'])

# Export to CSV
sorted_results = dh.sort_results(results)
dh.export_results(sorted_results)

# Creates the overall data histogram
overall_df = dh.overall_hist_prep(df)
vis.better_hist(overall_df)

# Create two line charts from the results
# All 20 items in one linechart is uninterpretable
res_wide = dh.pred_results_wide(results)
# Get the center of the columns
midpt = len(res_wide.columns) // 2
# Select everything to the left, then right of the midpoint respectively
res_wide_pt1 = res_wide.iloc[:, :midpt]
res_wide_pt2 = res_wide.iloc[:, midpt:]
vis.lineplot(res_wide_pt1)
vis.lineplot(res_wide_pt2)

# Create a heatmap of demand
heat_df = dh.heat_prep(results)
vis.heatmap(heat_df)

# Create a barchart of the overall APE score
mape_vis = dh.mape_prep(results)
vis.mape(mape_vis)