import copy
import os
import shelve
from scipy.spatial import distance
import warnings

from dimension_reduction import run_tsne
from dimension_reduction import euclidean_similiarty
import pandas as pd
import ast

class DataManager():
    """
    This class represents a whole dataset
    It can calculate and store the reduced 2D data
    """
    def __init__(self, config, x, y, kmeans, predictions, model):
        """

        Parameters
        ----------
        configuration of the model
        x da containing the attributes and values
        y dataset containing the label
        kmeans
        predictions
        """

        # Read every dataset and the kmeans object
        self.x = x
        self.y = y
        self.sens_attrs = config["sensitive"].split(",")
        # Remove the sensitive attributes
        # for sens in self.sens_attrs:
        #     self.x = self.x.loc[:, self.x.columns != sens]
        self.kmeans_data = self.x.copy()
        for sens in self.sens_attrs:
            self.kmeans_data = self.kmeans_data.loc[:, self.kmeans_data.columns != sens]

        self.kmeans = kmeans
        self.attributes = self.x.columns.tolist()
        # self.weight_dict = zmy_shelf["weight_dict"]
        self.cluster = self.kmeans.labels_
        # self.proxy = config["proxy"]
        self.tuning = config["hyperparameterTuning"]
        self.model = model
        # self.proxied_x_pred = self.proxy_dataset()
        self.tsne_data = None
        self.prediction_output = predictions



    def proxy_dataset(self):
        X_pred_cluster = copy.deepcopy(self.x)
        for attr in self.sens_attrs:
            X_pred_cluster = X_pred_cluster.loc[:, X_pred_cluster.columns != attr]

        if self.proxy in ("reweigh", "remove"):
            for col in list(X_pred_cluster.columns):
                if col in self.weight_dict:
                    X_pred_cluster[col] *= self.weight_dict[col]
                else:
                    X_pred_cluster = X_pred_cluster.loc[:, X_pred_cluster.columns != col]
        return X_pred_cluster

    def get_tsne_data(self):
        """
        This method returns the tsne data if not already done
        Returns
        -------
        """
        if self.tsne_data == None:
            x_test_copy = copy.deepcopy(self.x)
            self.tsne_data = run_tsne(x_test_copy)

        return self.tsne_data


    def get_plot_data(self, x_attr, y_attr):
        if (x_attr == y_attr == "tsne"):
            return self.calculate_plot_data_tsne()
        else:
            return self.calculate_plot_data(x_attr, y_attr)


    def calculate_plot_data(self, x_attr, y_attr):
        """
        This Method returns the data we want to plot for 2 specific attributes
        Returns the data in form: {x_attribute_value, y_attribute_value, cluster, predictedLabel, actualLabel, index}
        -------
        """
        plot_data = []
        for index, (real_index, point) in enumerate(self.x.iterrows()):
            # get the attribute values
            x_value = point[x_attr]
            y_value = point[y_attr]
            # get the cluster number
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
            cluster_number = self.kmeans.predict(self.kmeans_data.loc[real_index].values.reshape(1, -1))[0]
            # get the predicted label
            if self.tuning:
                predicted_label = self.prediction_output.loc[real_index][self.model + "_tuned"]
            else:
                predicted_label = self.prediction_output.loc[real_index][self.model]
            # get the actual label
            actual_label = self.y.iloc[index].iloc[0]


            datapoint = {"x": float(x_value), "y": float(y_value), "cluster": int(cluster_number),
                              "actualLabel": int(actual_label), "predictedLabel": int(predicted_label),
                              "index": real_index}

            # get the sensitve Attribute values
            for sens_attr in self.sens_attrs:
                datapoint[sens_attr] = int(self.prediction_output.loc[real_index][sens_attr])

            plot_data.append(datapoint)
        return plot_data

    def calculate_plot_data_tsne(self):
        """
         This Method returns the data we want to plot in the frontend if method tsne is selected.
        Returns the data in form: {x_coordinate, y_coordinate, cluster, predictedLabel, actualLabel, index}
        -------
        """
        self.get_tsne_data()

        plot_data = []
        for index, (real_index, point) in enumerate(self.x.iterrows()):
            # get the tsne data
            tsne_data = self.tsne_data[index]
            # get the cluster number
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
            cluster_number = self.kmeans.predict(self.kmeans_data.loc[real_index].values.reshape(1, -1))[0]
            # get the predicted label
            if self.tuning:
                predicted_label = self.prediction_output.loc[real_index][self.model + "_tuned"]
            else:
                predicted_label = self.prediction_output.loc[real_index][self.model]
            # get the actual label
            actual_label = self.y.iloc[index].iloc[0]

            datapoint = {"x": float(tsne_data[0]), "y": float(tsne_data[1]), "cluster": int(cluster_number),
             "actualLabel": int(actual_label), "predictedLabel": int(predicted_label),
              "index": real_index}

            # get the sensitve attributes
            for sens_attr in self.sens_attrs:
                datapoint[sens_attr] = int(self.prediction_output.loc[real_index][sens_attr])
            plot_data.append(datapoint)


        return plot_data




    def get_data_point(self, index):
        return self.x.loc[index]

    def get_proxied_data_point(self, index):
        return self.proxied_x_pred.loc[index]



    def get_counter_factual(self, index):
        """
        Iterates over every point in the dataset and returns the one that is the nearest and has a
        different predicted label
        Parameters
        ----------
        index of the datapoint you want to get the counterfactual point

        Returns the counterfactual point
        -------

        """

        ref_point = self.x.loc[index].values
        if self.tuning:
            ref_label = self.prediction_output.loc[index][self.model + "_tuned"]
        else:
            ref_label = self.prediction_output.loc[index][self.model]

        # filter every point that has the different label
        # go over every point and search for the most similiar one
        min_dist = float('inf')
        nearest_neighbor = None



        for current_index in self.x.index:
            if self.tuning:
                predicted_label = self.prediction_output.loc[current_index][self.model + "_tuned"]
            else:
                predicted_label = self.prediction_output.loc[current_index][self.model]

            if predicted_label != ref_label:
                if current_index != index:
                    current_point = self.x.loc[current_index].values
                    current_dist = euclidean_similiarty(ref_point, current_point)
                    if(current_dist < min_dist):
                        min_dist = current_dist
                        nearest_neighbor = current_index

        if nearest_neighbor:
            if self.tuning:
                predicted_label = self.prediction_output.loc[nearest_neighbor][self.model + "_tuned"]
            else:
                predicted_label = self.prediction_output.loc[nearest_neighbor][self.model]
            actual_label = self.y.loc[nearest_neighbor].iloc[0]
            cluster_number = self.kmeans.predict(self.kmeans_data.loc[nearest_neighbor].values.reshape(1, -1))[0]

            return{
                'index': int(nearest_neighbor),
                'predictedLabel': int(predicted_label),
                'actualLabel': int(actual_label),
                'cluster': int(cluster_number),
                'distance': float(min_dist),
            }
        else:
            return None