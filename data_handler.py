import os
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

class data_handler:

    # Gets the path the files reside in then marries it to the csv
    # This means CSV must be in same path as the .py files
    def load_data(self):
        # Agnostic path to the CSV file
        print("Getting demand")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(base_dir, 'demand_data.csv')

        # using this as a development file
        #data_file = os.path.join(base_dir, 'dev_demand.csv')

        # Read the data
        df = pd.read_csv(data_file)
        print("Data Loaded")
        return df
    
    def load_forecast(self):
        print("Getting Forecast File")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(base_dir, 'demand_forecast.csv')

        df = pd.read_csv(data_file)
        print("Forecast Loaded")
        return df
    
    # Removes negative values that are likely the result of returns or massive adjustments
    # Also sorts (so I don't have to in Excel)
    def clean_data(self, df):
        
        # Printing row counts in case it does some catastrophic level of cleaning
        print(f"Dirty Row Count: {len(df)}")

        # Drop nulls
        df.dropna(inplace=True)

        # Currently throwing out zero demand if it exists
        # This could be a user option
        demand_column = 'Original Demand Quantity'
        if demand_column in df.columns:
            df_cleaned = df.loc[(df['Original Demand Quantity'] > 0)]
        else:
            df_cleaned = df
        
        # Sorting by Demand Date descending
        df_cleaned = df_cleaned.sort_values(by='Demand Date', ascending=False)

        print(f"Clean Row Count: {len(df_cleaned)}")
        
        return df_cleaned
    
    # There is a lot of demand data available which isn't necessarily a benefit here
    # This will cut that window of time down to the years specified
    def trim_demand(self, df, years):

        # Turn Demand Date to a datetime object
        df['Demand Date'] = pd.to_datetime(df['Demand Date'], format='%m/%d/%y')

        # Determine the most recent date
        max_date = df['Demand Date'].max()
        print(f"The most recent date found is {max_date.date()}.")

        # Subtract the years value
        cutoff_date = max_date - pd.DateOffset(years=years)
        print(f"Data beginning at {cutoff_date.date()} will be kept.")

        # Trim
        df_trimmed = df.loc[df['Demand Date'] >= cutoff_date].copy()

        print(f"Trimmed Row Count: {len(df_trimmed)}")

        return df_trimmed

    # Moved all the featurizing here
    def data_wrangler(self, df):

        # Convert Demand Date to a datetime object
        df['Demand Date'] = pd.to_datetime(df['Demand Date'], format='%m/%d/%y')

        # Break up Demand Date into a number of different date features
        df['day_of_week'] = df['Demand Date'].dt.dayofweek
        #df['day_of_year'] = df['Demand Date'].dt.dayofyear
        df['month'] = df['Demand Date'].dt.month
        df['week_of_year'] = df['Demand Date'].dt.isocalendar().week
        df['year'] = df['Demand Date'].dt.year

        # I'm proud of this one. Creates a static "time index" the model likes.
        # This will ensure that any future predictions for dates that haven't happened will be perfectly at scale to training and testing data
        y2k = pd.to_datetime('01/01/00')
        df['time_index'] = (df['Demand Date'] - y2k).dt.days

        # List of the features
        features = ['Warehouse', 'Item', 'time_index', 'year', 'month', 'day_of_week', 'week_of_year']

        return features

    def oldest_demand_date(self, df):
        today = pd.to_datetime('today').year
        df['Demand Date'] = pd.to_datetime(df['Demand Date'], format='%m/%d/%y')
        dates = df['Demand Date'].min().year
        range = int(today - dates)
        return dates, range

    # Exports the CSV of the final predictions
    def export_results(self, results):
        results.to_csv('demand_predictions.csv', index=False)
        return
    
    def export_top(self, results):
        results.to_csv('top.csv', index=False)
        return
    
    # Combines the different columns to produce a final set of results 
    #def prep_results(self, X_test, y_test, y_test_alt, y_pred, y_pred_alt):
    def prep_results(self, X_test, y_test, y_pred):
        results = X_test.copy()
        results['Actual Demand'] = y_test
        #results['Revised Demand'] = y_test_alt
        results['Predicted Demand'] = y_pred
        #results['Predicted Revised Demand'] = y_pred_alt
        return results
    
    def prep_pred(self, X, y):
        results = X.copy()
        results['Predicted Demand'] = y
        return results
    
    # Used to morph dataframe for the line chart of overall data
    def pred_results_wide(self, df):
        wide = df.pivot_table(index='year', columns='Item', values='Actual Demand', aggfunc='sum', fill_value=0)
        wide.head()
        return wide
    
    # Used to select the columns used for the histogram of demand
    def overall_hist_prep(self, df):
        only_keep = ['Item', 'year', 'Original Demand Quantity']
        trimmed_df = df[only_keep]
        return df
    
    # Used to create the dataframe used for the heatmap
    def heat_prep(self, df):
        wide = df.pivot_table(index='Item', columns='year', values='Actual Demand', aggfunc='sum', fill_value=0)
        wide.head()
        return wide
    
    def item_selection_helper(self, df):
        # Get the mean and standard deviation
        item_detail = df.groupby('Item')['Original Demand Quantity'].agg(['mean', 'std']).reset_index()
        # Sort
        sorted_items = item_detail.sort_values(by='std', ascending=True)
        # Get top contenders
        winners_circle = sorted_items.head(20)['Item'].tolist()
        winners_df = df[df['Item'].isin(winners_circle)]
        return winners_df
    
    def mape(self, actual_dv, pred_dv):
        y = actual_dv
        y_pred = pred_dv

        mape = mean_absolute_percentage_error(y, y_pred)
        return mape
    
    def mape_prep(self, results):
        only_keep = ['Item', 'year', 'MAPE']
        trimmed_results = results[only_keep]
        sorted_results = trimmed_results.sort_values(by=['MAPE', 'year', 'Item'], ascending=[True, True, True])
        return sorted_results
    
    def sort_results(self, results):
        results = results.sort_values(by=['time_index', 'Warehouse', 'Item'], ascending=[False, True, True])
        return results


    