import math
import sys
import os
import subprocess
import json
import shelve
import warnings
import joblib
import numpy as np
import pandas as pd
from numpy.random.mtrand import random
from data_manager import DataManager
from dimension_reduction import run_tsne
import copy
import csv
import algorithm
import os
from aif360.datasets import BinaryLabelDataset
import random

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)
path = os.path.join(parent_parent_dir, "FairCR")
sys.path.append(path)


class Run:
    """
    Thiss class represents one configuration, the models and the results
    """

    def __init__(self, args, result_path):
        self.configuration = args
        # remove whitespaces before and after the name
        self.name = args["name"].strip()
        self.model = args["models"]
        self.ds = args["dataset"]
        self.testsize = args["testsize"]
        self.index = args["index"]
        self.sensitive = args["sensitive"].split(",")
        self.label = args["label"]
        self.metric = args["metric"]
        self.lam = args["lamda"]
        self.local_lam = args["localLamda"]
        self.favored = args["favored"]
        self.mem = args["memory"]
        self.time = args["runTime"]
        # self.predsize = args["predsize"]
        self.randomstate = args["randomstate"]
        self.tuning = args["hyperparameterTuning"]
        # how often do we want to train this configuration
        self.number_of_runs = int(args["numberOfRuns"])
        # path to the directory containg the datasets
        self.main_directory_path = os.path.dirname(os.path.abspath(__file__)) + "/"
        self.absolut_main_directory_path = os.path.abspath(self.main_directory_path)

        # get the attributes in the dataset
        csv_file = self.main_directory_path + "Datasets/" + self.ds + ".csv"
        attributes_pandas = pd.read_csv(csv_file).columns
        self.attributes = [attr for attr in attributes_pandas.tolist() if attr != self.index]
        # Initialize result
        self.prediction_result = None
        self.accuracy_result = None
        self.tsne_data = None
        self.absolut_link = result_path

        # storing and manipulating the datasets / tsne Data for every different model
        self.data_manager = {}

    def train_predict_evaluate(self):
        """
        Starts the offline phase then the online phase and then the evaluation of the Run
        -------
        """
        if not os.path.exists(self.absolut_link):
            os.makedirs(self.absolut_link)

        self.offlineAndOnline()
        self.evaluate()

        print("evaluation done ")

    def init_data_manager(self, model):
        if model in self.model:
            x_pred, y_pred = self.get_data_set()
            kmeans = self.get_kmeans()
            if self.tuning:
                csv_file = os.path.join(self.absolut_link, model + "_tuned_prediction.csv")
            else:
                csv_file = os.path.join(self.absolut_link, model + "_prediction.csv")
            predictions = pd.read_csv(csv_file, index_col=0)
            self.data_manager[model] = DataManager(self.configuration, x_pred, y_pred, kmeans, predictions, model)


    def get_plot_data(self, model_name, xAttribute, yAttribute):
        # check if the datamanager is already intialized
        if not model_name in self.data_manager:
            self.init_data_manager(model_name)
        modal_data_manager = self.data_manager[model_name]
        return modal_data_manager.get_plot_data(xAttribute, yAttribute)


    def offlineAndOnline(self):
        script = self.main_directory_path + "main.py"
        print(self.absolut_link)
        subprocess.check_call(
            [sys.executable, '-Wignore', script, '--output', str(self.absolut_link), '-skriptsDirectory',
             str(self.absolut_main_directory_path),
             '--ds', str(self.ds), '--sensitive', str(self.sensitive), '--favored', str(self.favored),
             '--label', str(self.label), '--testsize', str(self.testsize), '--randomstate', str(self.randomstate),
             '--models', str(self.model), '--metric', str(self.metric), '--tuning', str(self.tuning),
             '--lam', str(self.lam), '--local_lam', str(self.local_lam)])



    def evaluate(self):
        """
        starts the evaluation phase of the FALCC framework
        Parameters
        ----------
        model_list: name of every model that should be evaluated
        folder: The path to the folder containg the prediction results
        -------
        """
        model_list_eval = []
        for model in self.model:
            if self.tuning:
                model_list_eval.append(model + "_tuned")
            else:
                model_list_eval.append(model)


        evaluate_script = self.main_directory_path + "evaluation.py"

        subprocess.check_call(
            [sys.executable, '-Wignore', evaluate_script, '--folder', str(self.absolut_link), '-skriptsDirectory',
             str(self.absolut_main_directory_path),
             '--ds', str(self.ds), '--sensitive', str(self.sensitive), '--favored', str(self.favored),
             '--label', str(self.label), '--models', str(model_list_eval), '--metric', str(self.metric),
             '--name', 'EVALUATION'])

        print("evaluate done")

        # Also store the calculated results in the summary directory for the recommendations

        df = pd.read_csv("./Flask_Server/Summary/General_Results.csv")
        local_metric_dict = {
            "demographic_parity": "lrd_dp",
            "equalized_odds": "lrd_eod",
            "equal_opportunity": "lrd_eop",
            "treatment_equality": "lrd_te",
            "consistency": "consistency",
        }
        csv = pd.read_csv(self.absolut_link + "EVALUATION_" + str(self.ds) + ".csv", index_col="model")
        for model in self.model:
            res = df.loc[(df['model'] == model)
                         & (df['dataset'] == self.ds)
                         & (df['metric'] == self.metric)
                         & (df['tuning'] == self.tuning)
                         & (df['lambda'] == float(self.lam))
                         & (df['local_lambda'] == float(self.local_lam))]

            if res.empty:
                pos = len(df)
                df.at[pos, 'model'] = model
                df.at[pos, 'dataset'] = self.ds
                df.at[pos, 'metric'] = self.metric
                df.at[pos, 'tuning'] = self.tuning
                df.at[pos, 'lambda'] = float(self.lam)
                df.at[pos, 'local_lambda'] = float(self.local_lam)
                df.at[pos, 'count'] = 1
                df.at[pos, 'error_rate'] = csv.at[model, 'error_rate']
                df.at[pos, 'global'] = csv.at[model, self.metric]
                df.at[pos, 'local'] = csv.at[model, local_metric_dict[self.metric]]
                df.at[pos, 'runtime'] = csv.at[model, 'runtime']

            else:

                for i, row in res.iterrows():
                    counter = df.loc[i, 'count']
                    df.loc[i, 'error_rate'] = round(
                        (csv.at[model, 'error_rate'] + df.loc[i, 'error_rate'] * counter) / (counter + 1), 3)
                    df.loc[i, 'global'] = round(
                        (csv.at[model, self.metric] + df.loc[i, 'global'] * counter) / (counter + 1), 3)
                    df.loc[i, 'local'] = round(
                        (csv.at[model, local_metric_dict[self.metric]] + df.loc[i, 'local'] * counter) / (counter + 1),
                        3)
                    df.loc[i, 'runtime'] = round(csv.at[model, 'runtime'], 3)
                    df.loc[i, 'count'] += 1




        df.to_csv("./Flask_Server/Summary/General_Results.csv", index=False)


    def get_data_set(self):
        """
        Returns the evaluation dataset before proxy mitigation
        -------
        """
        filename = os.path.join(os.path.join(self.absolut_link), "shelve.out")
        # filename = self.absolut_link + "shelve.out"
        zmy_shelf = shelve.open(filename)
        X_test = zmy_shelf["x_test"]
        y_test = zmy_shelf["y_test"]
        zmy_shelf.close()
        return X_test, y_test


    def get_kmeans(self):
        filename = os.path.join(os.path.join(self.absolut_link), "shelve.out")
        # filename = self.absolut_link + "shelve.out"
        zmy_shelf = shelve.open(filename)
        kmeans = zmy_shelf["kmeans"]
        zmy_shelf.close()
        return kmeans


    def get_data_set_cluster(self, sens):
        """
        Returns the evaluation dataset used for the cluster algorithm
        -------
        Boolean sens: shell the dataset iclude the senstive attributes
        """
        filename = os.path.join(os.path.join(self.absolut_link), "shelve.out")
        # filename = self.absolut_link + "shelve.out"
        zmy_shelf = shelve.open(filename)
        # get the X_test and Y_test dataset
        if sens:
            X_test = zmy_shelf["X_test_cluster_with_sens"]
        else:
            X_test = zmy_shelf["X_test_cluster"]
            X_test.drop("cluster", axis=1, inplace=True)

        y_test = zmy_shelf["y_test"]
        zmy_shelf.close()
        return X_test, y_test


    def set_tsne_data(self, data):
        self.tsne_data = data


    def has_tsne_data(self):
        return self.tsne_data != None


    def tsne_point_to_data_point(self, index):
        """
        This functions receives the index of a tsne datapoint [x,y] and it outputs the coresponding full datapoint from the training data set
        Returns
        -------
        """
        x_test, y_test = self.get_data_set_cluster(True)
        datapoint = x_test.iloc[index]
        return datapoint


    def tsne_point_to_data_point_pre_proxy(self, index):
        """
        Determines for an index the attribute values before proxy discrimiantion mitigation was applied
        Parameters
        ----------
        index: the index of the datapoint
        Returns
        -------
        """
        x_text, y_text = self.get_data_set()
        datapoint = x_text.iloc[index]
        return datapoint

    def get_dataframe_from_point(self, datapoint):
        """
        This functions takes a dictionary reporesenting a datapoint and transforms it to a dataframe with the attributes
        in the right order
        Parameters
        ----------
        datapoint

        Returns dataframe
        -------
        """
        # filename = self.absolut_link + "shelve.out"
        filename = os.path.join(os.path.join(self.absolut_link), "shelve.out")
        zmy_shelf = shelve.open(filename)
        X_test_cluster = zmy_shelf["X_test_cluster"]
        zmy_shelf.close()
        x = pd.DataFrame([datapoint])
        X_test_cluster = X_test_cluster.drop(columns=["cluster"])
        x = x[X_test_cluster.columns]
        return x



    def get_all_models(self):
        """
        Returns all submodels used in this run
        if sbt = True this will be a list of lists
        if sbt = False this will just be a list
        -------
        """
        # filename = self.absolut_link + "shelve.out"
        filename = os.path.join(os.path.join(self.absolut_link), "shelve.out")
        zmy_shelf = shelve.open(filename)
        if self.sbt:
            model_list = zmy_shelf["model_list_sbt"]
            zmy_shelf.close()
            return model_list
        else:
            model_list = zmy_shelf["model_list"]
            zmy_shelf.close()
            return model_list


    def get_all_models_shortend(self):
        """
        Determines the name of every submodel for the falcc model
        Returns a list containg the shortend names of the submodels
        -------

        """
        model_list = []
        if not self.sbt:
            for model in self.get_all_models():
                model_list.append(os.path.splitext(os.path.basename(model))[0])
        else:
            for model_path in self.get_all_models()[0]:
                name = os.path.basename(model_path)
                sensitive_group, group_name = name.split('_', 1)
                group_name = group_name.rsplit('.pkl', 1)[0]
                model_list.append(group_name)

        return model_list


    def predict_datapoint(self, datapoint):
        """
        Determines the prediction of every model for a datapoint.
        Parameters
        ----------
        datapoint: the datapoint you want to predict
        Returns the predicted Label (0 or 1)
        -------
        """
        filename = os.path.join(self.absolut_link, "shelve.out")
        my_shelf = shelve.open(filename)
        predicitons = {}
        feature_order = my_shelf["x_test"].columns.tolist()

        for model in self.model:
            clf = my_shelf[model]
            if clf is not None:
                try:
                    dataframe = pd.DataFrame([datapoint])
                    # the same order as the fit method
                    dataframe = dataframe[feature_order]
                    predicitons[model] = clf.predict(dataframe)[0]
                except Exception as e:
                    print(f"Error predicting with model {model}: {e}")
                    predicitons[model] = -1
            else:
                predicitons[model] = -1
        my_shelf.close()
        return predicitons



    def get_clusters(self):
        filename = os.path.join(os.path.join(self.absolut_link), "cluster.out")
        zmy_shelf = shelve.open(filename)
        kmeans = zmy_shelf["kmeans"]
        zmy_shelf.close()
        return np.unique(kmeans.labels_).tolist()


    def get_cluster_information(self, cluster_number):
        """
        Parameters
        ----------
        cluster_number the local region we want to get the information
        Returns the inaccuracy.csv file for the specific cluster
        -------
        """

        if self.sbt:
            file_path = os.path.join(os.path.join(self.absolut_link),
                                     str(cluster_number) + "_inaccuracy_testphase_sbt.csv")
        else:
            file_path = os.path.join(os.path.join(self.absolut_link), str(cluster_number) + "_inaccuracy_testphase.csv")

        dataframe = pd.read_csv(file_path)
        return dataframe



