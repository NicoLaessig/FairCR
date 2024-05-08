from .FGLM_files.models import FairnessConstraintModel

class FairnessConstraintModelClass():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 c,
                 tau,
                 mu,
                 eps):
        """
        Args:
        """
        self.df_dict = df_dict
        self.c = c
        self.tau = tau
        self.mu = mu
        self.eps=eps


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        sens_idx = X_train.columns.get_loc(self.df_dict["sens_attrs"][0])
        self.model = FairnessConstraintModel(sensitive_index=sens_idx, c=self.c, tau=self.tau, mu=self.mu, eps=self.eps)
        self.model.fit(X_train.to_numpy(), y_train.to_numpy().reshape((y_train.to_numpy().shape[0],)))

        return self


    def predict(self, X_test):
        """
        Information

        Args:

        Returns:
        """
        pred = self.model.predict(X_test.to_numpy())

        return pred
       