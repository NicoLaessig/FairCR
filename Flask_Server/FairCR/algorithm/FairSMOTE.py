import copy
import pandas as pd
from .FairSMOTE_files.SMOTE import smote
from .FairSMOTE_files.Generate_Samples import generate_samples

class FairSMOTE():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 classifier,
                 remove):
        """
        Args:
        """
        self.df_dict = df_dict
        self.classifier = classifier
        self.remove = remove


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        label = list(y_train.columns)[0]
        train_df = copy.deepcopy(X_train)
        train_df[label] = copy.deepcopy(y_train)
        dataset_orig_train = copy.deepcopy(train_df)

        dict_cols = dict()
        cols = list(dataset_orig_train.columns)
        for i, col in enumerate(cols):
            dict_cols[i] = col

        lists = [[0, 1]]
        for sens in self.df_dict["sens_attrs"]:
            sens_vals = []
            for val in pd.unique(X_train[[sens]].values.ravel()):
                sens_vals.append(val)
            lists.append(sens_vals)

        groups = [label] + self.df_dict["sens_attrs"]
        grouped_train = dataset_orig_train.groupby(groups)
        groups_length = []
        for key, value in grouped_train:
            groups_length.append(len(grouped_train.get_group(key)))

        max_val = max(groups_length)
        count = 0
        for key, value in grouped_train:
            gdf = grouped_train.get_group(key)
            if len(gdf) < max_val:
                increase = max_val - len(gdf)
                gdf_bal = generate_samples(increase,gdf,'',dict_cols)
            else:
                gdf_bal = copy.deepcopy(gdf)
            if count == 0:
                df_new = copy.deepcopy(gdf_bal)
            else:
                df_new = df_new.append(gdf_bal)
            count += 1

        X_train, y_train = df_new.loc[:, df_new.columns != label], df_new[label]
        if self.remove:
            for sens in self.df_dict["sens_attrs"]:
                X_train = X_train.drop(sens, axis=1)
        self.classifier.fit(X_train, y_train)

        return self


    def predict(self, X_test):
        """
        Information

        Args:

        Returns:
        """
        if self.remove:
            for sens in self.df_dict["sens_attrs"]:
                X_test = X_test.drop(sens, axis=1)

        pred = self.classifier.predict(X_test)

        return pred
