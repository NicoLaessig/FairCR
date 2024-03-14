import pandas as pd
from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.datasets import BinaryLabelDataset

class AIF_MetaFairClassifier():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 metric,
                 tau):
        """
        Args:
        """
        self.df_dict = df_dict
        if metric == "demographic_parity":
            self.metric = "sr"
        else:
            self.metric = "fdr"
        self.tau = tau


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
        dataset_train = BinaryLabelDataset(df=train_df, label_names=[self.df_dict["label"]], protected_attribute_names=self.df_dict["sens_attrs"])
        self.model = MetaFairClassifier(tau=self.tau, sensitive_attr=self.df_dict["sens_attrs"][0], type=self.metric)
        self.model.fit(dataset_train)

        return self


    def predict(self, X_test):
        """
        Information

        Args:

        Returns:
        """
        #test_df = pd.merge(X_test, y_test, left_index=True, right_index=True)
        X_test[self.df_dict["label"]] = 0
        dataset_test = BinaryLabelDataset(df=X_test, label_names=[self.df_dict["label"]], protected_attribute_names=self.df_dict["sens_attrs"])
        pred = list(self.model.predict(dataset_test).labels.ravel())

        return pred
