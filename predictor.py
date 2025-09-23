import joblib

class predictor:
    
    # Generates the predictions based on independent variables passed to it
    def crystal_ball(self, model, input_data):
        y_pred = model.predict(input_data)
        return y_pred