from .GradualCompatibility_files.lr import CustomLogisticRegression

class GradualCompatibility():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 reg,
                 reg_val,
                 weights_init,
                 lam):
        """
        Args:
        """
        self.df_dict = df_dict
        self.reg = reg
        self.reg_val = reg_val
        if weights_init == "None":
            self.weights_init = None
        else:
            self.weights_init = weights_init
        self.lam = lam


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        A_train = X_train[self.df_dict["sens_attrs"]].values
        X_train = X_train.drop(self.df_dict["sens_attrs"], axis=1).values

        alpha = 0
        beta = 0
        gamma = 0
        if self.reg in (1, 4, 5, 7):
            alpha = self.reg_val
        elif self.reg in (2, 4, 6, 7):
            beta = self.reg_val
        elif self.reg in (3, 5, 6, 7):
            gamma = self.reg_val

        self.model = CustomLogisticRegression(X=X_train, Y=y_train.to_numpy().reshape((y_train.to_numpy().shape[0],)), A=A_train.reshape(-1,),
            weights_init=self.weights_init, alpha=alpha, beta=beta, gamma=gamma, _lambda=self.lam)
        self.model.fit()

        return self


    def predict(self, X_test):
        """
        Information

        Args:

        Returns:
        """
        X_test = X_test.drop(self.df_dict["sens_attrs"], axis=1).values

        pred = []
        prediction = self.model.predict_prob(X_test)
        for i, pr in enumerate(prediction):
            if pr > 0.5:
                pred.append(1)
            else:
                pred.append(0)

        return pred
        