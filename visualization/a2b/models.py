# -*- coding: utf-8 -*-
"""
Created on 12.06.24

@author: Katja

"""
import warnings

import numpy as np
import torch
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn


class AnthroToBeta:

    def __init__(self, model_path, model_type="svr"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.gender = "neutral" if "neutral" in model_path else "female" if "female" in model_path else "male" if "male" in model_path else None
        assert self.gender is not None, "Gender needs to be contained in model path"
        self.num_betas = 16 if self.gender == "neutral" else 10 if self.gender == "female" else 11
        if model_type == "svr":
            self.model = AnthroToBetaSVR()
            self.model.load_weights(model_path)
        elif model_type == "nn":
            self.model = AnthroToBetaNN(self.num_betas)
            self.model.load_weights(model_path)
            self.model = self.model.to(self.device)
        else:
            raise ValueError(f"Model type {model_type} not supported")

    def format_input(self, anthro_params):
        if self.model_type == "svr" and isinstance(anthro_params, torch.Tensor):
            return anthro_params.cpu().numpy()
        elif self.model_type == "svr":
            return anthro_params
        elif isinstance(anthro_params, np.ndarray):
            return torch.from_numpy(anthro_params).float().to(self.device)
        else:
            return anthro_params.to(self.device)

    def format_output(self, output, as_tensor=False):
        if self.model_type == "svr":
            return torch.from_numpy(output).to(self.device) if as_tensor else output
        else:
            return output.detach().cpu().numpy() if not as_tensor else output

    def predict(self, anthro_params, as_tensor=False):
        anthro_params = self.format_input(anthro_params)
        result = self.model.forward(anthro_params)
        return self.format_output(result, as_tensor)


class AnthroToBetaWrapper:

    def forward(self, anthro):
        raise NotImplementedError

    def load_weights(self, weights_path):
        raise NotImplementedError


class AnthroToBetaNN(nn.Module, AnthroToBetaWrapper):

    def __init__(self, num_betas):
        super().__init__()

        layers = [nn.Linear(36, 330), nn.Tanh(),
                  nn.Linear(330, 330), nn.Tanh(),
                  nn.Linear(330, 330), nn.Tanh(),
                  nn.Linear(330, 330), nn.Tanh(),
                  nn.Linear(330, num_betas)]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        initialization = lambda l: nn.init.xavier_uniform_(l, gain=nn.init.calculate_gain('tanh'))
        for l in layers[::2]:
            initialization(l.weight)

        self.layers = nn.Sequential(*layers)

    def load_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path, map_location=self.device))

    def forward(self, anthro):
        return self.layers(anthro)


class AnthroToBetaSVR(AnthroToBetaWrapper):

    def __init__(self, estimator=None):
        self.regressor = make_pipeline(StandardScaler(),
                         MultiOutputRegressor(SVR(C=3790.63, epsilon=0.012, kernel="rbf")))

        self.estimator = estimator

    def fit(self, anthros, betas):
        self.estimator = self.regressor.fit(anthros, betas)

    def load_weights(self, weights_path):
        # The models are trained with scikit-learn 1.3.2, but it also works with 1.6.1, but we don't want the warnings
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        self.estimator = torch.load(weights_path, weights_only=False)

    def forward(self, anthros):
        assert self.estimator is not None, "You need to train or set the model first"
        return self.estimator.predict(anthros)