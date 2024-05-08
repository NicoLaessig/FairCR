import pandas as pd
import tensorflow as tf
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.datasets import BinaryLabelDataset

class AIF_AdversarialDebiasing():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 classifier,
                 adversary_loss,
                 debias):
        """
        Args:
        """
        self.df_dict = df_dict
        self.classifier = classifier
        self.adversary_loss = adversary_loss
        self.debias = debias


    def close_sess(self):
        """
        ...
        """
        self.sess.close()
        tf.compat.v1.reset_default_graph()


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
        dataset_train = BinaryLabelDataset(df=train_df, label_names=[self.df_dict["label"]], protected_attribute_names=self.df_dict["sens_attrs"])
        tf.compat.v1.disable_eager_execution()
        self.sess = tf.compat.v1.Session()
        self.model = AdversarialDebiasing(self.df_dict["unprivileged_groups"], self.df_dict["privileged_groups"], scope_name=self.classifier, debias=self.debias, sess=self.sess, adversary_loss_weight=self.adversary_loss)
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
  