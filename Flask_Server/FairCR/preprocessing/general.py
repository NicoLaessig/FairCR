from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import ClusterCentroids
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import copy

class GeneralPreprocessing():
    """
    information [...]

    References:
        ...
    """
    def __init__(self, X_train, y_train, sens_attrs, favored, label):
        """
        Args:
        """
        self.X_train = X_train
        self.y_train = y_train
        self.sens_attrs = sens_attrs
        self.favored = favored
        self.label = label


    def binarize(self):
        """
        Information

        Args:

        Returns:
        """
        for i, row in self.X_train.iterrows():
            fav = True
            for j, sens in enumerate(self.sens_attrs):
                if row[sens] != self.favored[j]:
                    fav = False
                    self.X_train.loc[i, "sensitive"] = 0
                    break
            if fav:
                self.X_train.loc[i, "sensitive"] = 1
        for sens in self.sens_attrs:
            self.X_train = self.X_train.drop(sens, axis=1)
        self.sens_attrs = ["sensitive"]
        self.favored = (1)

        return self.X_train, self.sens_attrs, self.favored


    def balance(self, method, balance_type):
        """
        ...
        """
        ###FIX###
        if method == "Tomek":
            strat = SMOTETomek()
        elif method == "ADASYN":
            strat = ADASYN()
        elif method == "ENN":
            strat = SMOTEENN()
        elif method == "ClusterCentroids":
            strat = ClusterCentroids()
        #Others are also possible

        if balance_type == "classic":
            self.X_train, self.y_train = strat.fit_resample(self.X_train, self.y_train)
        elif balance_type == "adapted":
            df = self.X_train.merge(self.y_train, left_index=True, right_index=True, how="inner")
            key_list = []
            cols = copy.deepcopy(self.sens_attrs)
            cols.append(self.label)
            grouped_df = df.groupby(cols)
            count = 0
            for key, item in grouped_df:
                part_df = grouped_df.get_group(key)
                for i, row in part_df.iterrows():
                    df.loc[i, "target"] = count
                count += 1
                key_list.append(key)

            X_train = df.loc[:, df.columns != "target"]
            y_train = df["target"]
            for col in cols:
                X_train = X_train.drop(col, axis=1)

            X_train_new, y_train_new = strat.fit_resample(X_train, y_train)
            df_new = X_train_new.merge(y_train_new, left_index=True, right_index=True, how="inner")

            grouped_df = df_new.groupby("target")
            for i in range(count):
                part_df = grouped_df.get_group(i)
                for r, row in part_df.iterrows():
                    for j, col in enumerate(cols):
                        df_new.loc[r, col] = key_list[i][j]

            df_new = df_new.drop("target", axis=1)

            self.X_train = df_new.loc[:, df_new.columns != self.label]
            self.y_train = df_new[self.label]

        return self.X_train, self.y_train


    def feature_selection(self, method, X_test):
        """
        ...
        """
        cols = list(self.X_train.columns)
        if method == "VarianceThreshold":
            selector = VarianceThreshold()
            selector.fit(self.X_train)
            variances = selector.variances_
            rm_list = []
            for i, col in enumerate(cols):
                if col not in self.sens_attrs and variances[i] < 0.1:
                    rm_list.append(col)
            
        elif method == "RFECV":
            rfe = RFECV(LogisticRegression(), cv=10)
            rfe.fit(self.X_train, self.y_train)
            support = rfe.support_
            rm_list = []
            for i, col in enumerate(cols):
                if col not in self.sens_attrs and not support[i]:
                    rm_list.append(col)

        #Now remove the unimportant features
        if len(rm_list) >= 1:
            for rm in rm_list:
                self.X_train = self.X_train.loc[:, self.X_train.columns != rm]
                X_test = X_test.loc[:, X_test.columns != rm]

        return self.X_train, X_test


    def dimensionality_reduction(self, X_test):
        """
        ...
        """
        prot_feats_train = []
        prot_feats_test = []
        X_train_no_prot = copy.deepcopy(self.X_train)
        X_test_no_prot = copy.deepcopy(X_test)
        for sens in self.sens_attrs:
            prot_feats_train.append(self.X_train[sens])
            prot_feats_test.append(X_test[sens])
            X_train_no_prot = X_train_no_prot.loc[:, X_train_no_prot.columns != sens]
            X_test_no_prot = X_test_no_prot.loc[:, X_test_no_prot.columns != sens]
        pca = PCA(n_components="mle")
        pca.fit(X_train_no_prot)
        X_train_np = pca.transform(X_train_no_prot)
        X_test_np = pca.transform(X_test_no_prot)
        cols = [str(i) for i in range(len(X_train_np[0]))]
        X_train = pd.DataFrame(X_train_np, columns=cols)
        X_test = pd.DataFrame(X_test_np, columns=cols)

        for i, sens in enumerate(self.sens_attrs):
            X_train[sens] = prot_feats_train[i].tolist()
            X_test[sens] = prot_feats_test[i].tolist()

        X_train["index"] = prot_feats_train[0].index.values.tolist()
        X_test["index"] = prot_feats_test[0].index.values.tolist()
        self.X_train = X_train.set_index("index", inplace=True)
        X_test = X_test.set_index("index", inplace=True)

        return self.X_train, X_test
