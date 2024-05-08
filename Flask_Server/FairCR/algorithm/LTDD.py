#Added optional protected attribute removal
from scipy import stats

class LTDD():
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
        column_train = list(X_train.columns)
        self.slope_store = []
        self.intercept_store = []
        self.column_u = []

        for i in column_train:
            if i != self.df_dict["sens_attrs"][0]:
                slope, intercept, rvalue, pvalue, stderr = stats.linregress(X_train[self.df_dict["sens_attrs"][0]], X_train[i])
                if pvalue < 0.05:
                    self.column_u.append(i)
                    self.slope_store.append(slope)
                    self.intercept_store.append(intercept)
                    X_train[i] = X_train[i] - (X_train[self.df_dict["sens_attrs"][0]] * slope + intercept)

        if self.remove:
            self.X_train = X_train.drop([self.df_dict["sens_attrs"][0]], axis=1)
        else:
            self.X_train = X_train
        self.y_train = y_train

        self.classifier.fit(self.X_train, self.y_train)

        return self


    def predict(self, X_test):
        """
        Information

        Args:

        Returns:
        """
        for i in range(len(self.column_u)):
            X_test[self.column_u[i]] = X_test[self.column_u[i]] - (X_test[self.df_dict["sens_attrs"][0]] * self.slope_store[i] + self.intercept_store[i])

        if self.remove:
            X_test = X_test.drop([self.df_dict["sens_attrs"][0]], axis=1)

        pred = self.classifier.predict(X_test)

        return pred
