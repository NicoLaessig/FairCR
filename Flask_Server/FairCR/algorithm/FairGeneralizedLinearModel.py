from .FGLM_files.models import FairGeneralizedLinearModel

class FairGeneralizedLinearModelClass():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 lam,
                 family,
                 discretization):
        """
        Args:
        """
        self.df_dict = df_dict
        self.lam = lam
        self.family = family
        self.discretization = discretization


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        sens_idx = X_train.columns.get_loc(self.df_dict["sens_attrs"][0])
        self.model = FairGeneralizedLinearModel(sensitive_index=sens_idx, lam=self.lam, family=self.family, discretization=self.discretization)
        self.model.fit(X_train.to_numpy(), y_train.to_numpy().reshape((y_train.to_numpy().shape[0],)))

        return self


    def predict(self, X_test):
        """
        Information

        Args:

        Returns:
        """
        pred = self.model._predict(X_test.to_numpy())

        return pred
        