from sklearn.metrics import mean_squared_error

class CustomLoss:
    def __call__(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
