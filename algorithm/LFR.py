"""
TODO
"""
#1. Conversion of X_train to dataset_train
#2. do_eval in main.py
#3. LFR is implemented differently, using custom classifier for fit/predict
import pandas as pd
from aif360.algorithms.preprocessing import LFR
from aif360.datasets import BinaryLabelDataset


class AIF_LFR():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 classifier,
                 k,
                 Ax,
                 Ay,
                 Az,
                 remove):
        """
        Args:
        """
        self.df_dict = df_dict
        self.classifier = classifier
        self.k = k
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az
        self.remove = remove


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
        dataset_train = BinaryLabelDataset(df=train_df, label_names=[self.df_dict["label"]], protected_attribute_names=self.df_dict["sens_attrs"])
        self.model = LFR(self.df_dict["unprivileged_groups"], self.df_dict["privileged_groups"], k=self.k, Ax=self.Ax, Ay=self.Ay, Az=self.Az)
        self.model = self.model.fit(dataset_train)
        dataset_transf_train = self.model.transform(dataset_train)
        dataset_transf_train = dataset_transf_train.convert_to_dataframe()[0]
        self.X_train = dataset_transf_train.loc[:, dataset_transf_train.columns != self.df_dict["label"]]
        self.y_train = dataset_transf_train[self.df_dict["label"]]
        if self.remove:
            for sens in self.df_dict["sens_attrs"]:
                self.X_train = self.X_train.drop(sens, axis=1)

        #In case that every label is set to the same value
        label_vals = self.y_train.unique()
        if len(label_vals) == 1:
            self.same_label = True
            self.label_val = label_vals[0]
        else:
            self.same_label = False
            self.classifier.fit(self.X_train, self.y_train)

        return self


    def predict(self, X_test):
        """
        Information

        Args:

        Returns:
        """
        if self.same_label:
            pred = [self.label_val for i in range(len(X_test))]
        else:
            if self.remove:
                for sens in self.df_dict["sens_attrs"]:
                    X_test = X_test.drop(sens, axis=1)

            pred = self.classifier.predict(X_test)

        return pred
