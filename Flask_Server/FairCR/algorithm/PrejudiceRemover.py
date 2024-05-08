import pandas as pd
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.datasets import BinaryLabelDataset

class AIF_PrejudiceRemover():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 eta):
        """
        Args:
        """
        self.df_dict = df_dict
        self.eta = eta


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
        dataset_train = BinaryLabelDataset(df=train_df, label_names=[self.df_dict["label"]], protected_attribute_names=self.df_dict["sens_attrs"])
        self.model = PrejudiceRemover(eta=self.eta, sensitive_attr=self.df_dict["sens_attrs"][0], class_attr=self.df_dict["label"])
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
