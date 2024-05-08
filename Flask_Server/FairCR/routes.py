import math

import pandas as pd
from flask import Flask, jsonify, request
import json

import traceback
from werkzeug.utils import secure_filename
from recommender import generate_general_recommendation_plots
from multiple_runs import MultipleRuns
from run_manager import RunManager
import os


class Router:
    """
    The Router class is responsible for managing the routing of HTTP requests to various endpoints within a Flask application.
    """

    # Object containing all the information about the runs
    run_manager = None
    # object containg data for the file upload
    datasets_folder = os.path.abspath("Flask_Server/FairCR/Datasets")
    ALLOWED_EXTENSIONS = {'csv'}

    def __init__(self, app):
        """
        Set up route handlers
        Parameters
        ----------
        app: An instance of the Flask application.
        """

        @app.route('/run-analyzer/init-runs', methods=['POST'])
        def init_runs():
            """
            Initialize the run configurations based on the input data provided by the user.
            Extracts the configuration for each different configuration and intializes the runs / multipleRuns / runManager Objects
            It checks for every configuration if there exist a already trained version

            Returns A JSON response with the status of initialization, including lists of trained and not trained runs.
            """

            runs = []
            # Access the different configurations
            for run_arguments in request.get_json():
                # initialize for every configuration a multiple_run object
                multiple_run = MultipleRuns(run_arguments)
                # store it in an array
                runs.append(multiple_run)
            # create the run manager object for all multiple_runs
            self.run_manager = RunManager(runs)
            # get all not trained Runs
            not_trained_runs, trained_runs = self.run_manager.get_not_trained_run_names()
            print("init done")
            return jsonify({'status': "inProgress", 'notTrainedRuns': not_trained_runs, 'trainedRuns': trained_runs})

        @app.route('/run-analyzer/train-runs', methods=['POST'])
        def train_runs():
            """
            Triggers the training process, prediction and evaluation process of the FALCC-Framework.
            It does this for every "new" run or if the user specifially told to retrain an old run
            Returns A JSON response containing the training status, run names, all cluster numbers, sensitivite attributes and results
            for every multiple_Run object !
            -------
            """
            if self.run_manager == None:
                return {"status": "error"}
            else:
                # this data maps the "old" runs to a boolean value indicating if they should be retrained
                self.run_manager.trained_run_informations = request.json.get("retrainInformations")
                print("start train")
                self.run_manager.train_evaluate_all()

                # Get the recommendations for the first run
                # recommendations = self.run_manager.dataset_specific_recommendation()
                # if recommendations is not None:
                #     recommendations_dict = recommendations.to_dict(orient='records')
                # else:
                #     recommendations_dict = []
                # Construct the Result dictionary
                response_data = {
                    "status": "inProgress",
                    "runNames": self.run_manager.get_run_names(),
                    "modelNames": self.run_manager.get_model_names(),
                    "cluster": self.run_manager.get_cluster_numbers(),
                    "sensitive": self.run_manager.get_sensitive(),
                    "results": self.run_manager.results,
                    # "recommendations": recommendations_dict
                }
                # Return the dictionary as JSON response
                return jsonify(response_data)

        @app.route('/run-analyzer/inputdata/attributes', methods=['GET'])
        def getDataForAttributes():
            """
            Retrieves data for specific attributes related to a given run name.
            The final data contains the values for every datapoint for the both attributes, the index, the senstive group,
            the cluster number, the predicted label and the actual label
            Returns:
                A JSON response containing the data for the requested attributes.
            """
            attribute1 = request.args.get('attribute1')
            attribute2 = request.args.get('attribute2')
            run_name = request.args.get('runName')
            model_name = request.args.get('modelName')
            plot_data = self.run_manager.get_data_for_run(run_name, model_name, attribute1, attribute2)
            response_data = {
                "plotData": plot_data,
            }
            return jsonify(response_data)

        @app.route('/run-analyzer/inputdata/getattributes', methods=['GET'])
        def getAttributesForRun():
            """
            Fetches the list of attributes for the dataset a given run name was trained on
            Returns:
                    A JSON response with a list of attributes related to the specified run.
            """
            run_name = request.args.get('runName')
            attributes = self.run_manager.get_attributes_for_run(run_name)
            attributes = ["tsne"] + attributes
            response_data = {
                "attributes": attributes
            }
            return jsonify(response_data)

        @app.route('/run-analyzer/inputdata/index', methods=['GET'])
        def getDatpointForRun():
            """
            Retrieves a specific tupel for a given index for a specific run
            Returns:
                A JSON response with the tupel (after and before proxy discrimnination mitigation)
            """
            index = int(request.args.get('index'))
            run_name = request.args.get('runName')
            model_name = request.args.get('modelName')



            datapoints = self.run_manager.get_datapoint_for_run(model_name, run_name, index)

            if datapoints is not None:
                print("datapoint = ", datapoints)
                response_data = {
                    "status": "success",
                    #"datapoint_proxied": datapoints[0],
                    "datapoint_before": datapoints
                }
            else:
                response_data = {
                    "status": "failure"
                }

            return jsonify(response_data)


        @app.route('/run-analyzer/specificrecommendations', methods=['GET'])
        def get_specific_recommendations():

            run_name = request.args.get('runName')
            recommendations = self.run_manager.dataset_specific_recommendation(run_name)

            if recommendations is not None:
                recommendations_dict = recommendations.to_dict(orient='records')
                # replace every nan value with -1
                # Iterate over each dictionary in the list
                for item in recommendations_dict:
                    # Now iterate over each key in the dictionary
                    for key in item.keys():
                        # Check if the value is NaN and replace it
                        if isinstance(item[key], (int, float)) and math.isnan(item[key]):
                            item[key] = -1
            else:
                recommendations_dict = []
            response = {"recommendations": recommendations_dict}

            return jsonify(response)


        @app.route('/recommendations/generalrecommendations', methods=['GET'])
        def get_general_recommendations():
            metrics = [request.args.get('metric')]
            tuning_str = request.args.get('tuning')
            if tuning_str == "false":
                tuning = False
            else:
                tuning = True
            lam = float(request.args.get('lambda'))
            local_lam = float(request.args.get('localLambda'))
            datasize = int(request.args.get('datasize'))
            mem = float(request.args.get('mem'))
            time = float(request.args.get('time'))
            general_results = "./Flask_Server/Summary/General_Results.csv"
            general_mem_results = "./Flask_Server/Summary/General_Memory_Results.csv"
            if tuning:
                general_time_results = "./Flask_Server/Summary/General_Runtime_Results.csv"
            else:
                general_time_results = "./Flask_Server/Summary/General_Runtime_Iteration_Results.csv"

            try:
                recommendations = generate_general_recommendation_plots(metrics, tuning, lam, local_lam, datasize,
                                                                       mem, time, general_results, general_mem_results, general_time_results)[0]
            except Exception as e:
                print(type(e))
                print(e.args)
                print(e)
                traceback.print_exc()
                recommendations = None

            if recommendations is not None:
                recommendations_dict = recommendations.to_dict(orient='records')
                # replace every nan value with -1
                # Iterate over each dictionary in the list
                for item in recommendations_dict:
                    # Now iterate over each key in the dictionary
                    for key in item.keys():
                        # Check if the value is NaN and replace it
                        if isinstance(item[key], (int, float)) and math.isnan(item[key]):
                            item[key] = -1
            else:
                recommendations_dict = []
            response = {"recommendations": recommendations_dict}
            return jsonify(response)
        @app.route('/run-analyzer/inputdata/predict', methods=['POST'])
        def predict_datapoint():
            """
            Predicts label for a given datapoint across all possible multible_run objects AND their submodels
            Returns
                A JSON response containing the predictions
            """
            data = request.get_json()
            datapoint = data["datapoint"]


            run_name = data["runName"]
            # This dictionary contains for every run the prediction
            # eg: Run1 -> 0 , Run2 -> 1, Run3 -> 1
            all_predictions = self.run_manager.predict_with_all_falcc_models(datapoint, run_name)
            # We also want for every Run to get the predictions of every submodel
            # eg: Run1 -> {AdaboostModel1 -> 1, AdaboostModel2 -> 1}, Run2 -> {AdaboostModel3 -> 0} ...
            results = {
                "run_predictions": all_predictions,
            }
            print("results = ", results)
            return results

        @app.route('/run-analyzer/inputdata/trainedruns', methods=['GET'])
        def get_trained_runs():
            """
            Retrieves all configurations that have been trained and are stored in the modal storage
            Returns
                A JSON reponse with the configuration of all trained runs
            """
            main_directory_path = "Flask_Server/FairCR/Results/"
            main_directory = os.path.abspath(main_directory_path)
            trained_runs = {}
            for subdir, dirs, files in os.walk(main_directory):
                if "config.json" in files:
                    absolute_path_to_json = os.path.abspath(os.path.join(subdir, "config.json"))
                    with open(absolute_path_to_json, 'r') as json_file:
                        try:
                            data = json.load(json_file)
                            if isinstance(data, dict):
                                config = data
                                name = config["name"]
                                del config["name"]
                                trained_runs[name] = config
                        except json.decoder.JSONDecodeError:
                            print("wrong structure of json file")

            return jsonify(trained_runs)

        @app.route('/run-analyzer/inputdata/counterfactual', methods=['GET'])
        def get_counterfactual():
            """
            Retrieves counterfactual data for a given run based on the provided index.

            Returns:
                A JSON response containing the counterfactual data.
            """
            index = int(request.args.get('index'))
            run_name = request.args.get('runName')
            modelname = request.args.get('modelName')

            data = self.run_manager.get_counter_factual(index, run_name, modelname)
            return jsonify(data)

        @app.route('/run-analyzer/inputdata/clusterInformation', methods=['GET'])
        def get_cluster_information():
            """
            Fetches the information created in the modal assessment phase of the FALCC Framework about a specific Cluster
            Returns:
                A JSON response with cluster information for the specified run.
            """
            cluster = int(request.args.get('cluster'))
            run_name = request.args.get('runName')
            data = self.run_manager.get_cluster_information(cluster, run_name)
            return data


        @app.route("/run-analyzer/uploadDataset", methods=['POST'])
        def upload_file():
            if 'file' not in request.files:
                return 'No Data in the Request', 400
            file = request.files['file']
            if file.filename == '':
                return 'No file selected', 400
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(self.datasets_folder, filename)
                file.save(file_path)

                # store the dataset also in the allDatasets.json file
                # Erhalte zusätzliche Daten
                index = request.form.get('index')
                label = request.form.get('label')
                protectedAttributes = request.form.get('sensitive')
                favoredAttributes = request.form.get('favored')
                update_json(filename, index, label, protectedAttributes, favoredAttributes)
                dataset_names, dataset_details = get_dataset_names()
                return jsonify({
                    "datasetNames": dataset_names,
                    "datasetDetails": dataset_details
                 }), 200
            else:
                return 'Invalid file format', 400

        @app.route("/run-analyzer/getDatasetNames", methods=["GET"])
        def send_dataset_names():
            dataset_names, dataset_details = get_dataset_names()
            return jsonify({
                "datasetNames": dataset_names,
                "datasetDetails": dataset_details
            })



        def allowed_file(filename):
            """Überprüft, ob die Dateiendung erlaubt ist."""
            return '.' in filename and \
                   filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS

        def get_dataset_names():
            JSON_FILE_PATH = os.path.join(self.datasets_folder, 'allDatasets.json')
            with open(JSON_FILE_PATH, 'r') as f:
                data = json.load(f)

            # Erstellen eines Arrays mit allen Datensatznamen
            dataset_names = list(data.keys())

            # Erstellen eines Dictionaries, das jeden Namen auf seine Attribute abbildet
            dataset_details = {name: data[name] for name in dataset_names}

            return dataset_names, dataset_details

        def update_json(filename, index, label, protectedAttributes, favoredAttributes):
            json_path = os.path.join(self.datasets_folder, 'allDatasets.json')
            with open(json_path, 'r+') as f:
                data = json.load(f)
                new_data = {
                    filename: {
                        "index": index,
                        "label": label,
                        "protectedAttributes": protectedAttributes,
                        "FavoredAttributes": favoredAttributes
                    }
                }
                data.update(new_data)
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()