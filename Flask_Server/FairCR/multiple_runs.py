import os
from run import Run
import json
import pandas as pd

class MultipleRuns:
    """
    This Class stores n Runs (specified by the user).
    Every Run is trained on the SAME configuration.
    """

    def __init__(self, args):
        self.configuration = args
        self.run_list = []
        self.name = str(args["name"])
        # path containing every result data and model data
        main_directory = os.path.dirname(os.path.abspath(__file__))
        self.result_path = os.path.join(main_directory, os.path.join("Results", self.name), "")


    def add_already_trained_runs(self, result_path):
        """
        Goes over the given result folder and adds every trained Run to the run_list
        Parameters
        ----------
        result_path the result folder
        """
        for directory in os.listdir(result_path):
            sub_result_path = os.path.join(result_path, directory)
            if os.path.isdir(sub_result_path):
                self.run_list.append(Run(self.configuration, sub_result_path))
        self.next_run_number = len(self.run_list)


    def train_multiple(self):
        """
        Trains a specified amount of runs for this configuration.
        Write the new amount of runs and the configuration in a file
        get the evaluation for every Run (average and combined)
        Returns
        -------
        """
        # Create the directory if neccesary
        if not os.path.isdir(self.result_path):
            try:
                os.makedirs(self.result_path)
            except OSError as e:
                print(f"Error: Failed to create directory '{self.result_path}':{e}")

        # Add all already trained old runs
        self.add_already_trained_runs(self.result_path)

        # add / train the specified number of new runs
        count = int(self.configuration["numberOfRuns"])
        for run_number in range(count):
            result_sub_path = os.path.join(self.result_path, str(self.next_run_number), "")
            new_run = Run(self.configuration, result_sub_path)
            # Start offline, online, evaluation
            new_run.train_predict_evaluate()
            self.next_run_number = self.next_run_number + 1
            self.run_list.append(new_run)


        # Write the configuration to a json file
        self.write_config_to_json(len(self.run_list))

        # get all evaluation files combined
        self.get_all_evaluations()




    def write_config_to_json(self, number_of_trained_runs):
        """
        Write the configuration to a file in the result folder
        Returns
        -------
        """

        print("number of trained Runs = ", number_of_trained_runs)

        file_path = os.path.join(self.result_path, "config.json")
        # If the file does already exist -> there are already trained models for this run
        self.configuration["numberOfRuns"] = number_of_trained_runs

        # Write the new configurations into the json file
        with open(file_path, 'w') as json_file:
            json.dump(self.configuration, json_file)




    def get_all_evaluations(self):
        """
        This Method iterates over every trained run with the same configuration an
        writes down every EVALUATION.csv file into 1 Single EVALUATION file.
        Also creates EVALUATION_AVG.csv which contains the average values for every metric
        Returns
        -------
        """
        combined_df = pd.DataFrame()
        for run in self.run_list:
            sub_result_path = run.absolut_link

            evaluation_path = os.path.join(sub_result_path, "EVALUATION_" + self.configuration["dataset"] + ".CSV")
            if os.path.exists(evaluation_path):
                eval_df = pd.read_csv(evaluation_path)
                combined_df = pd.concat([combined_df, eval_df], ignore_index=True)

        print("combined Dataframe = " , combined_df)
        combined_df.to_csv(os.path.join(self.result_path, "EVALUATION_COMBINED.csv"), index=False, mode='w')
        average_df = combined_df.groupby(['model']).mean(numeric_only=True).reset_index()
        average_df.to_csv(os.path.join(self.result_path, "EVALUATION_AVG.csv"), index=False, mode='w')



    def get_proxied_datapoint(self, model_name, index):
        # datapoint_proxied = self.run_list[0].data_manager.get_proxied_data_point(index)
        datapoint_before = self.run_list[0].data_manager[model_name].get_data_point(index)
        # return datapoint_proxied.to_dict(), datapoint_before.to_dict()
        return datapoint_before.to_dict()

