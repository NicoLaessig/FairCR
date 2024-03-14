import copy
import numpy as np
from algorithm.JiangNachum_files.label_bias import LabelBiasDP_helper, LabelBiasEOD_helper, LabelBiasEOP_helper

class JiangNachum():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 classifier,
                 metric,
                 estimators,
                 learning_rate,
                 remove):
        """
        Args:
        """
        self.df_dict = df_dict
        self.classifier = classifier
        self.metric = metric
        if metric == "demographic_parity":
            self.LB = LabelBiasDP_helper()
        elif metric == "equalized_odds":
            self.LB = LabelBiasEOD_helper()
        elif metric == "equal_opportunity":
            self.LB = LabelBiasEOP_helper()
        else:
            self.LB = LabelBiasDP_helper()
        self.estimators = estimators
        self.lr = learning_rate
        self.remove = remove


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        protected_train = [np.array(X_train[g]) for g in self.df_dict["sens_attrs"]]
        multipliers = np.zeros(len(protected_train))
        if self.metric == "equalized_odds":
            multipliers = np.zeros(len(protected_train) * 2)
        weights = np.array([1] * X_train.shape[0])

        if self.remove:
            for sens in self.df_dict["sens_attrs"]:
                X_train = X_train.drop(sens, axis=1)
        X_tr_np = X_train.to_numpy()
        y_tr_np = y_train.to_numpy().reshape((y_train.to_numpy().shape[0],))

        for it in range(self.estimators):
            if self.metric not in ("equalized_odds", "equal_opportunity"):
                weights = self.LB.debias_weights(y_tr_np, protected_train, multipliers)
            clf = copy.deepcopy(self.classifier)
            clf.fit(X_tr_np, y_tr_np, weights)
            prediction = clf.predict(X_tr_np)
            if self.metric in ("equalized_odds", "equal_opportunity"):
                weights = self.LB.debias_weights(y_tr_np, prediction, protected_train, multipliers)

            acc, violations, pairwise_violations = self.LB.get_error_and_violations(prediction, y_tr_np, protected_train)
            multipliers += self.lr * np.array(violations)

        self.classifier.fit(X_train, y_train, weights)

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
