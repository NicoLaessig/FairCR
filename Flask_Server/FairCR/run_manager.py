import csv
import math
import warnings

import numpy as np
import pandas as pd
import json
import shutil
from run import Run
from multiple_runs import MultipleRuns
import os
from recommender import dataset_specific_recommendation

class RunManager():
    """
    Manages multiple configurations for fair classification models, overseeing the training and evaluation of multiple runs per configuration.
    Attributes:
        runs (list[MultipleRuns]): A list of MultipleRuns objects, each representing a configuration and the associated trained runs.
        results (dict): After evaluation, contains the results for each run for the performance measures (e.g., accuracy.
        trained_run_informations (dict): Maps previous (already trained) configurations to a boolean indicating whether they should be retrained.
        falcc_run (Run): The first run using the FALCC algorithm found among all runs, or None if not present.
    """

    runs: [MultipleRuns] = []
    results: {} = {}
    trained_run_informations: {} = {}


    def __init__(self, runs):
        """
        Initializes the RunManager with a list of MultipleRuns objects.
        Parameters:
            runs (list[MultipleRuns]): A list of runs to manage.
        """
        self.main_result_path = os.path.abspath("Flask_Server/FairCR/Results/")
        self.runs = runs


    def train_evaluate_all(self):
        """
        Trains and evaluates all managed runs.
        Checks for similar existing runs to avoid retraining identical configurations
        unless its specified by the user.
        """
        for run in self.runs:
            # check if there is a similiar run
            similiar_run = self.exist_similiar_run(run.configuration)
            # When there is no similiar run just train it
            if similiar_run == None:
                run.train_multiple()
            # When there is a similiar
            else:
                # Check if the user wants to train
                # The User wants to retrain
                if self.trained_run_informations[run.name] == True:
                    # special case the user wants to use the "old" model results but gave the "new" run a different name
                    if run.result_path != similiar_run:
                        # Change the result folder path from the new run to the old one
                        run.result_path = similiar_run
                        # And then train the run
                        run.train_multiple()
                    else:
                        run.train_multiple()
                # The user does not want to retrain
                else:
                    # special case the user wants to use a "old" configuraiton but gave the new Run a different name
                    if run.result_path != similiar_run:
                        # Change the result folder path form the new run to the old BUT DONT TRAIN
                        run.result_path = similiar_run
                        # add the old runs to the runList
                        run.add_already_trained_runs(similiar_run)

        self.results = self.read_evaluations_csv()

    def read_evaluations_csv(self):
        """
        Reads evaluation results from 'EVALUATION_AVG.csv' and 'EVALUATION_COMBINED.csv' for each run. Constructs a performance
        summary for every run and submodel.

        Returns:
            tuple: A dictionary mapping every Run to the performance of its submodels and to the finals performance
        """
        # Dictionary mapping every Run to the dictionaries of every model
        # eg {Run1 -> {AVG(accuracy) : 80%, AVG(accuracy): 60%}, Run2 -> ... }
        # and a Dictionary mapping every Run to more dictionaries for every sub run
        # eg {Run1 -> [{accuracy1 : 80%, fairness: 60%}, {accuracy2 : 82%, fairness: 61%} , ... ], Run2 -> ..., }
        results_AVG = {}
        results_COMB = {}

        for run in self.runs:
            df_AVG = pd.read_csv(run.result_path + "/EVALUATION_AVG.csv")
            df_COMB = pd.read_csv(run.result_path + "/EVALUATION_COMBINED.csv")
            run_results = {}
            for index, row in df_AVG.iterrows():
                model_name = row["model"]
                # Dictionary mapping every metric key to the value eg : accuracy -> 80%
                metric_values = {col: row[col] for col in df_AVG.columns if col != 'model'}
                # Todo solve problem with NaN values
                metric_values = {k: (0 if pd.isna(v) else v) for k, v in metric_values.items()
                                 if not k.startswith('Unnamed')}
                run_results[model_name] = metric_values

            results_AVG[run.name] = run_results
            submodel_run_results = {}
            model_name_counter = {}
            for index, row in df_COMB.iterrows():
                metric_values = {col: row[col] for col in df_AVG.columns if col != 'model'}
                # Todo solve problem with NaN values
                metric_values = {k: (0 if pd.isna(v) else v) for k, v in metric_values.items()}
                if row["model"] in model_name_counter:
                    number = model_name_counter[row["model"]]
                    new_name = row["model"] + str(number)
                    model_name_counter[row["model"]] = number + 1
                else:
                    number = 0
                    new_name = row["model"] + str(number)
                    model_name_counter[row["model"]] = number + 1

                submodel_run_results[new_name] = (metric_values)

            results_COMB[run.name] = submodel_run_results


        return results_AVG, results_COMB



    def find_run_by_name(self, name):
        """
        Finds a run by its name.

        Parameters:
            name (str): The name of the run to find.

        Returns:
            Run: The run matching the given name, or None if not found.
        """
        for run in self.runs:
            run_name = run.name.strip()
            if run_name == name.strip():
                return run
        return None

    def get_datapoint_for_run(self, model_name, run_name, index):
        """
        Retrieves a specific datapoint for a given run name and index.

        Args:
            model_name (str): the name of the model
            run_name (str): The name of the run.
            index (int): The index of the datapoint.

        Returns:
            tuple: A tuple containing proxied and original datapoint, or None if the run is not found.
        """
        run = self.find_run_by_name(run_name)
        if (run != None):
            return run.get_proxied_datapoint(model_name,index)
        else:
            return None

    def get_not_trained_runs(self):
        """
        Identifies runs that have not been trained yet.

        Returns:
            tuple: A tuple containing lists of not trained and already trained runs.
        """
        not_trained_runs = []
        trained_runs = []
        for run in self.runs:
            if self.exist_similiar_run(run.configuration) == None:
                not_trained_runs.append(run)
            else:
                trained_runs.append(run)

        return not_trained_runs, trained_runs

    def get_not_trained_run_names(self):
        """
        Identifies runs that have not been trained yet.

        Returns:
            tuple: A tuple containing lists of not trained and already trained run names.
        """
        names_not_trained = []
        names_trained = []
        ntr, tr = self.get_not_trained_runs()
        for run in ntr:
            names_not_trained.append(run.name)
        for run in tr:
            names_trained.append(run.name)
        return names_not_trained, names_trained

    def get_run_names(self):
        """
        Retrieves the names of all managed runs.

        Returns:
            list[str]: A list of all run names.
        """
        run_names = []
        for run in self.runs:
            run_names.append(run.name)
        return run_names


    def get_model_names(self):
        """
        Retrieves all run names and all coressponding model names
        Returns
        -------
        """
        model_names = {}
        for run in self.runs:
            model_names[run.name] = run.run_list[0].model
        return model_names

    def get_cluster_numbers(self):
        """
        Collects cluster information from the first run of each configuration.

        Returns:
            dict: A dictionary mapping run names to their cluster numbers.
        """
        run_cluster_information = {}
        for run in self.runs:
            run_cluster_information[run.name] = run.run_list[0].get_clusters()
        return run_cluster_information

    def get_sensitive(self):
        """
        Gathers sensitive groups from the first run of each configuration.

        Returns:
            dict: A dictionary mapping run names to their sensitve groups.
        """
        sensitive = {}
        for run in self.runs:
            sensitive[run.name] = run.run_list[0].sensitive
        return sensitive

    def get_attributes_for_run(self, run_name):
        """
        Fetches attributes for the first run of a specific configuration.
        Args:
            run_name (str): The name of the run.

        Returns:
            list: A list of attributes for the specified run, or an empty list if the run is not found.
        """
        run = self.find_run_by_name(run_name).run_list[0]
        if run != None:
            return run.attributes
        else:
            return []

    def get_data_for_run(self, run_name, model_name, xAttribute, yAttribute):
        """
        Retrieves plot data for specific attributes of a given run.

        Parameters:
            run_name (str): The name of the run.
            xAttribute (str): The attribute for the x-axis.
            yAttribute (str): The attribute for the y-axis.

        Returns:
            list: A list of datapoints for plotting, or an empty list if the run is not found.
        """
        # get the run belonging to the name
        run = self.find_run_by_name(run_name).run_list[0]
        if run != None:
            data = run.get_plot_data(model_name, xAttribute, yAttribute)
            # data = run.prepare_plot_data(xAttribute, yAttribute)
            return data
        else:
            return []

    def get_counter_factual(self, index, run_name, model_name):
        """
        Fetches counterfactual data for a specific datapoint of a run.

        Parameters:
            index (int): The index of the datapoint.
            run_name (str): The name of the run.

        Returns:
            dict: The counterfactual data, or None if the run is not found.
        """
        run = self.find_run_by_name(run_name)
        if run != None:
            counterfactual_data = run.run_list[0].data_manager[model_name].get_counter_factual(index)
            return counterfactual_data
        else:
            return None

    def get_cluster_information(self, cluster, run_name):
        """
        Fetches the data from the model assessment phase for a specific cluster and run.

        Parameters:
            cluster (int): The cluster number.
            run_name (str): The name of the run.

        Returns:
            str: A JSON string containing the cluster information, or None if the run is not found.
        """
        def get_short_model_name(model_path):
            file_name = os.path.basename(model_path)
            short_name = '_'.join(file_name.split('_')[:-1])
            return short_name

        run = self.find_run_by_name(run_name)
        if run != None:
            cluster_information = run.run_list[0].get_cluster_information(cluster)
            cluster_information['model'] = cluster_information['model'].apply(get_short_model_name)
            json_data = cluster_information.to_json(orient='records')
            return json_data
        else:
            return None

    # def get_optimal_model_combination(self, run_name, cluster, weight, metric):
    #     """
    #     Determines the best model combination for a given cluster and metric of a run.
    #
    #     Parameters:
    #         run_name (str): The name of the run.
    #         cluster (int): The cluster number.
    #         weight (float): The weight for the metric.
    #         metric (str): The metric to consider.
    #
    #     Returns:
    #         list: A list of short names for the best model combination, or an empty list if the run is not found.
    #     """
    #     run = self.find_run_by_name(run_name)
    #     if run != None:
    #         modelComb, _ = run.run_list[0].get_optimal_model_combination(cluster, metric, weight)
    #     # shorten the result
    #     usedModels = []
    #     for model_path in modelComb:
    #         for element in model_path:
    #             file_name = os.path.basename(element)
    #             short_name = '_'.join(file_name.split('_')[:-1])
    #             usedModels.append(short_name)
    #
    #     return usedModels

    def get_runs_with_same_dataset(self, run_name):
        """
        Determines every configuration that has the same dataset
        Parameters
        ----------
        run_name: The name of the run

        Returns: list of runs with the same configuration as the run with the name run_name
        -------

        """
        run1 = self.find_run_by_name(run_name)
        dataset1 = run1.run_list[0].configuration["dataset"]
        result = [run1]
        for run2 in self.runs:
            if run1 != run2:
                dataset2 = run2.run_list[0].configuration["dataset"]
                if dataset1 == dataset2:
                    result.append(run2)

        return result



    def predict_with_all_falcc_models(self, datapoint, run_name):
        """
        This Method iterates over every run that can predict the same datapoint as the run with the name run_name.
        For every of this runs it returns the final prediction
        Parameters
        ----------
        datapoint: The datapoint you want to predict
        run_name: the run_name you want to use

        Returns a dictionary mapping each run name to its prediction
        eg {Run1 -> 0, Run2 -> 1}
        -------

        """
        same_runs = self.get_runs_with_same_dataset(run_name)
        all_predictions = {}
        for run in same_runs:
            run_name = run.name
            predicted_value = run.run_list[0].predict_datapoint(datapoint)
            all_predictions[run_name] = predicted_value
        return all_predictions

    def exist_similiar_run(self, run_config):
        """
        Determines if for a specific configuration an already trained run exists
        Parameters
        ----------
        run_config: The configuration you want to check

        Returns the link to the trained run or None if there exists None
        -------
        """
        for subdir, dirs, files in os.walk(self.main_result_path):
            if "config.json" in files:
                absolute_path_to_json = os.path.abspath(os.path.join(subdir, "config.json"))
                with open(absolute_path_to_json, 'r') as json_file:
                    data = json.load(json_file)
                    if self.compare_configs(data, run_config):
                        # There exist an already trained run with the exact same config
                        absolut_link = os.path.abspath(subdir)
                        return absolut_link
        return None

    def compare_configs(self, config1, config2):
        """
        Checks if two configurations are the same
        Parameters
        ----------
        config1: The first configuration
        config2: The second configuration

        Returns True if they are similiar
        -------

        """
        config1_copy = config1.copy()
        config2_copy = config2.copy()
        # ignored parameters
        keys_to_ignore = ["randomstate", "numberOfRuns", "name"]
        for key in keys_to_ignore:
            config1_copy.pop(key, None)
            config2_copy.pop(key, None)
        # Special handling for arrays in the configurations
        for key in config1_copy.keys():
            if isinstance(config1_copy[key], list) and isinstance(config2_copy[key], list):
                # Convert lists to sets to ignore order
                if set(config1_copy[key]) != set(config2_copy[key]):
                    return False
                # If the sets are equal, set the original lists to the same value to ensure
                # they do not cause inequality in the final comparison
                config1_copy[key] = config2_copy[key] = sorted(config1_copy[key])
            elif config1_copy[key] != config2_copy[key]:
                return False

        return True


    def dataset_specific_recommendation(self, run_name):

        run = self.find_run_by_name(run_name).run_list[0]
        models = run.model
        datasets = [run.ds]
        metrices = [run.metric]
        tuning = run.tuning
        lam = float(run.lam)
        local = float(run.local_lam)
        mem = float(run.mem)
        time = float(run.time)
        general_results = "./Flask_Server/Summary/General_Results.csv"
        general_mem_results = "./Flask_Server/Summary/General_Memory_Results.csv"
        if tuning:
            general_time_results = "./Flask_Server/Summary/General_Runtime_Results.csv"
        else:
            general_time_results = "./Flask_Server/Summary/General_Runtime_Iteration_Results.csv"
        dataset_name = run.ds
        dataset_path = run.main_directory_path + "/Datasets/" + dataset_name + ".csv"
        datasize = len(pd.read_csv(dataset_path))

        print(run, models, tuning, datasize)

        warnings.filterwarnings("ignore", category=np.RankWarning)

        try:
            recommendation = dataset_specific_recommendation(datasets[0], models, metrices, tuning, lam, local, mem,
                                                             time, general_results, general_mem_results, general_time_results, datasize)

            return recommendation[0]
        except Exception as e:
            print(type(e))
            print(e.args)
            print(e)
            return None