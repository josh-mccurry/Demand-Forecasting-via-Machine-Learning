from sklearn.linear_model import LinearRegression
import joblib

class modeler:

    def split_data(self, df, features):

        X = df[features]
        y = df['Original Demand Quantity']
        #y_alt = df['Revised Demand Quantity']

        # This sets the size of the split for the training data
        # Currently set to allocate 80% to train, 20% to test
        split = int(len(df) * 0.8)

        # Independent variables for train
        X_train = X[:split]
        # Dependent variables for train and train alt (i.e., Revised Demand Quantity - which is effectively a manual demand override)
        y_train = y[:split]
        #y_train_alt = y_alt[:split]

        # Independent variables for test
        X_test = X[split:]
        # Dependent variables for test and test alt
        y_test = y[split:]
        #y_test_alt = y_alt[split:]

        return X_train, X_test, y_train, y_test
    
    def featurize(self, df, features):
        X = df[features]
        return X
    
    # Trains the linear regression model
    def train_model(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    # Added to fulfil the "tools to maintain the product" requirement
    def load_model(self, path):
        joblib.load(path)
        return print("Model loaded")
    
    def save_model(self, model):
        joblib.dump(model, 'demand_forecast_model.joblib')
        return print("Model saved")