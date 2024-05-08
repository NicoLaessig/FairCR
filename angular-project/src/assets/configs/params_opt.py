import json

params = json.load(open('params.json'))
opt_param = dict()

models = ["DisparateImpactRemover",\
    "LFR",\
    "Reweighing",\
    "AdversarialDebiasing",\
    "GerryFairClassifier",\
    "MetaFairClassifier",\
    "PrejudiceRemover",\
    "ExponentiatedGradientReduction",\
    "GridSearchReduction",\
    "EqOddsPostprocessing",\
    "CalibratedEqOddsPostprocessing",\
    "RejectOptionClassification",\
    "Fair-SMOTE",\
    "FairSSL-LS",\
    "FairSSL-CT",\
    "FairSSL-LP",\
    "FairSSL-ST",\
    "JiangNachum",\
    "AdaFair",\
    "FairGeneralizedLinearModel",\
    #"LinearFERM",\
    "SquaredDifferenceFairLogistic",\
    "FairnessConstraintModel",\
    "DisparateMistreatmentModel",\
    "ConvexFrameworkModel",\
    "HSICLinearRegression",\
    "GeneralFairERM",\
    "FAGTB",\
    "FairDummies",\
    "HGR",\
    #"FairMixup",\
    "MultiAdversarialDebiasing",\
    "GradualCompatibility",\
    "FaX",\
    "DPAbstention",\
    "LTDD",\
    #"FL-BD",\
    #"FL-CB",\
    #"FL-JB",\
    #"FL-BTEO",\
    #"FL-Adv",\
    #"FL-EAdv",\
    #"FL-DAdv",\
    #"FL-AADv",\
    #"FL-Gate",\
    #"FL-FairBatch",\
    #"FL-FairSCL",\
    #"FL-EO",\
    #"FL-INLP",\
    #"FL-SoftGated",\
    #"FL-GBT",\
    #"FL-UKNN",\
    #"FL-ARL",\
    #"FairDomainAdaptation",\
    "GetFair",\
    "FairBayesDPP",\
    "iFair",\
    "FairBoost",\
    "FALCCClassic",\
    "LogisticRegression",\
    "DecisionTree",\
    "AdaBoost",\
    "RandomForest",\
    "SVM",\
    "MLP",\
    "kNN",\
    "LogisticRegressionRemoved",\
    "DecisionTreeRemoved",\
    "AdaBoostRemoved",\
    "RandomForestRemoved",\
    "SVMRemoved",\
    "MLPRemoved",\
    "kNNRemoved",\
    "Salimi",\
    "Zhang",\
    "Galhotra"
    ]

datasets = ["german", "communities", "compas", "credit_card_clients", "adult_data_set_sex",\
    "adult_data_set_race", "acs2017_census",\
    "implicit10", "implicit20", "implicit30", "implicit40", "implicit50",\
    "social10", "social20", "social30", "social40", "social50",\
    "social30_10p", "social30_20p", "social30_30p", "social30_40p", "compas_race", "compas_sex"]

states = [100, 1, 42, 500, 7000]

li = []
for model in models:
    paramlist = list(params[model]["default"].keys())
    opt_param[model] = dict()
    for ds in datasets:
        opt_param[model][ds] = dict()
        for s in states:
            opt_param[model][ds][s] = dict()
            for param in paramlist:
                opt_param[model][ds][s][param] = params[model]["default"][param]


json.dump(opt_param, open('params_opt.json', 'w'), indent=4)
json.dump(opt_param, open('params_opt_demographic_parity.json', 'w'), indent=4)
json.dump(opt_param, open('params_opt_equalized_odds.json', 'w'), indent=4)
json.dump(opt_param, open('params_opt_treatment_equality.json', 'w'), indent=4)
json.dump(opt_param, open('params_opt_consistency.json', 'w'), indent=4)