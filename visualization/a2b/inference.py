# -*- coding: utf-8 -*-
"""
Created on 17.06.24

@author: Katja

"""
import json
import os
import pickle

import numpy as np
import torch

from visualization.a2b.models import AnthroToBeta
from visualization.a2b.names import anthro_names


def read_pkl_measurements(file, reduction="median"):
    """
    Reads anthropometric measurements from a pickle file and reduces them to median values. Expects a dictionary with subject
    names as keys and a dictionary with anthropometric measurements as values. The dictionary has the anthropometric measurement
    names as keys and a list of values as values. The dictionary should also contain a key "original_median_betas" with the median
    beta values of the originally estimated betas. This entry is not reduced to median values
    :param file: path to file
    :param reduction: currently, only "median" is supported
    :return:
    """
    with open(file, "rb") as f:
        measurements = pickle.load(f)
    if reduction == "median":
        for s in measurements:
            orig_median = measurements[s]["original_median_betas"]
            measurements[s] = {name: np.median(val) for name, val in measurements[s].items() if name != "original_median_betas"}
            measurements[s]["original_median_betas"] = orig_median
    else:
        raise ValueError("Currently only median reduction is implemented")
    measure_list = {s: [measurements[s][name] for name in anthro_names] for s in measurements}
    median_betas = {s: measurements[s]["original_median_betas"] for s in measurements}
    return measure_list, median_betas

def read_json_measurements(file):
    """
    Reads measurements from a json file. The file should contain a dictionary mapping from subject names to a dict of
    anthropometric measurements (measurement name to value)
    :param file: json file path
    :return:
    """
    with open(file, "r") as f:
        measurements = json.load(f)
        measure_list = {s: [measurements[s][name] for name in anthro_names] for s in measurements}
        return measure_list, None

def inference_with_all_models(anthro_measurements_path, anthro_param_name, beta_param_save_path, model_files):
    """
    Inference for all models in the given model files with the given anthropometric measurements in the file
    :param anthro_measurements_path: path to anthropometric measurements. Can be either json or pkl. See requirements for file
    content in functions above
    :param anthro_param_name: Name of the source anthropometric measurements, will be used as key in the result dict
    :param beta_param_save_path: path to save the beta parameters, can be either json or pkl
    :param model_files: list of paths to saved a2b models. All models need to have "svr" or "nn" in their name in order to
    identify the type of model
    :return: nothing, but saves a dictionary with all beta parameters per subject. Dictionary structure:
        {"<anthro_param_name>_A2B_<model_type>_<gender>": {"gender": gender,
                                                            "name": <same_as_key>,
                                                            "num_betas": number of beta parameters used,
                                                            "<subject_name>": beta parameters}}
    """
    if anthro_measurements_path.endswith(".pkl"):
        anthro_measurements, median_betas = read_pkl_measurements(anthro_measurements_path, reduction="median")
    else:
        anthro_measurements, median_betas = read_json_measurements(anthro_measurements_path)
    all_beta_results = {}
    for model_file in model_files:
        print(f"Inference for Model: {model_file}")
        model_type = "svr" if "svr" in model_file else "nn"
        model = AnthroToBeta(model_file, model_type=model_type)
        gender = "neutral" if "neutral" in model_file else "female" if "female" in model_file else "male"
        beta_param_res = {
                "gender": gender,
                "name": f"{anthro_param_name}_A2B_{model_type}_{gender}"
                }
        for s in anthro_measurements:
            beta_param_res[s] = model.predict(torch.tensor(anthro_measurements[s]).float().unsqueeze(0))[0]
            beta_param_res["num_betas"] = beta_param_res[s].shape[0]
            if beta_param_save_path.endswith(".json"):
                beta_param_res[s] = beta_param_res[s].tolist()
        all_beta_results[beta_param_res["name"]] = beta_param_res

    if median_betas is not None:
        all_beta_results[anthro_param_name] = {
                "name":   anthro_param_name,
                "gender": "neutral",
                }
        for s in median_betas:
            median_bs = median_betas[s][0] if len(median_betas[s].shape) > 1 else median_betas[s]
            all_beta_results[anthro_param_name][s] = median_bs
            all_beta_results[anthro_param_name]["num_betas"] = median_betas[s].shape[-1]
            if beta_param_save_path.endswith(".json"):
                all_beta_results[anthro_param_name][s] = all_beta_results[anthro_param_name][s].tolist()
    save_dir = os.path.dirname(beta_param_save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)
    if beta_param_save_path.endswith(".pkl"):
        with open(beta_param_save_path, "wb") as f:
            pickle.dump(all_beta_results, f)
    else:
        with open(beta_param_save_path, "w") as f:
            json.dump(all_beta_results, f)


def get_betas(json_measurements, save_path):

    best_model_files = ["visualization/a2b/neutral_uniform_ext_nn.pth"]
    inference_with_all_models(json_measurements, "visualize", save_path, best_model_files)