import pandas as pd
from aif360.algorithms.inprocessing import GerryFairClassifier
from aif360.datasets import BinaryLabelDataset

class AIF_GerryFairClassifier():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 gamma):
        """
        Args:
        """
        self.df_dict = df_dict
        self.gamma = gamma
        #Currently only FP and FN is supported, unlike shown in the documentation
        self.metric = "FP"


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
        dataset_train = BinaryLabelDataset(df=train_df, label_names=[self.df_dict["label"]], protected_attribute_names=self.df_dict["sens_attrs"])
        self.model = GerryFairClassifier(gamma=self.gamma, fairness_def=self.metric)
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
        