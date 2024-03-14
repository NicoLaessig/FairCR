import pandas as pd
from aif360.algorithms.inprocessing import GridSearchReduction
from aif360.datasets import BinaryLabelDataset

class AIF_GridSearchReduction():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 classifier,
                 metric,
                 lam,
                 remove):
        """
        Args:
        """
        self.df_dict = df_dict
        self.classifier = classifier
        if metric == "demographic_parity":
            self.metric = "DemographicParity"
        elif metric in ("equalized_odds", "equal_opportunity"):
            self.metric = "EqualizedOdds"
        else:
            self.metric = "DemographicParity"
        self.lam = lam
        self.remove = remove


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        drop_prot_attr = bool(self.remove)

        train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
        dataset_train = BinaryLabelDataset(df=train_df, label_names=[self.df_dict["label"]], protected_attribute_names=self.df_dict["sens_attrs"])
        self.model = GridSearchReduction(self.classifier, self.metric, self.df_dict["sens_attrs"], constraint_weight=self.lam, drop_prot_attr = drop_prot_attr)
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
        