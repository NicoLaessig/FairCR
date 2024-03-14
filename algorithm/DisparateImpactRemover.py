#Added optional protected attribute removal

import pandas as pd
import numpy as np
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import BinaryLabelDataset

class AIF_DisparateImpactRemover():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 classifier,
                 repair,
                 remove):
        """
        Args:
        """
        self.df_dict = df_dict
        self.classifier = classifier
        self.repair = repair
        self.remove = remove


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
        dataset_train = BinaryLabelDataset(df=train_df, label_names=[self.df_dict["label"]], protected_attribute_names=self.df_dict["sens_attrs"])
        self.index = dataset_train.feature_names.index(self.df_dict["sens_attrs"][0])

        self.model = DisparateImpactRemover(repair_level=self.repair)

        train_repd = self.model.fit_transform(dataset_train)

        if self.remove:
            self.X_train = np.delete(train_repd.features, self.index, axis=1)
        else:
            self.X_train = train_repd.features

        self.y_train = train_repd.labels.ravel()

        self.classifier.fit(self.X_train, self.y_train)

        return self


    def predict(self, X_test):
        """
        Information

        Args:

        Returns:
        """
        if self.remove:
            X_test = X_test.drop(self.df_dict["sens_attrs"][0], axis=1)

        pred = self.classifier.predict(X_test.to_numpy())

        return pred
        