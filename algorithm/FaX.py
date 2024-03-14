from algorithm.FaX_files import FaX_methods

class FaX():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict):
        """
        Args:
        """
        self.df_dict = df_dict


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        X = X_train.loc[:, X_train.columns != self.df_dict["sens_attrs"][0]].to_numpy()
        Z = X_train[self.df_dict["sens_attrs"][0]].to_frame().to_numpy()
        Y = y_train.to_numpy()

        self.model = FaX_methods.MIM(X, Z, Y)

        return self


    def predict(self, X_test):
        """
        Information

        Args:

        Returns:
        """
        X = X_test.loc[:, X_test.columns != self.df_dict["sens_attrs"][0]].to_numpy()
        pred = self.model.predict(X)

        return pred
