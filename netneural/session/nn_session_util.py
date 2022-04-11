import json
import lzma
import pickle
from datetime import datetime

import numpy as np

from ..network.feature_scaling import StandardScaler, NormalScaler
from ..network.nn import NeuralNetwork
from ..network.one_hot_encoder import OneHotEncoder
from ..pca.pca import PCA


def load_from_config(file_name):
    if file_name.endswith('.json'):
        with open(file_name) as json_file:
            data = json.load(json_file)
    else:
        with lzma.open(file_name, "rb") as fin:
            data = pickle.load(fin)
    try:
        regression = data['regression']
    except KeyError:
        regression = False
    try:
        pca_n = data['pca_n']
    except KeyError:
        pca_n = None
    try:
        unique_labels = data['unique_labels']
        encoder = OneHotEncoder(unique_labels=unique_labels)
    except KeyError:
        encoder = None
    weights = []
    try:
        scaler_name = data['scaler']
        if scaler_name == 'standard':
            scaler = StandardScaler(data['standard_scaler_mean'], data['standard_scaler_std'])
        else:
            scaler = NormalScaler(data['normal_scaler_min'], data['normal_scaler_max'])
    except KeyError:
        scaler = None
    for weight in data['weights']:
        weights.append(np.array(weight))
    neural_network = NeuralNetwork(tuple(data['shape']),
                                   activation_function=data['activation_function'],
                                   weight_matrices=weights,
                                   encoder=encoder,
                                   regression=regression,
                                   scaler=scaler)
    if pca_n is not None:
        pca = PCA(data['pca_n'], np.array(data['eigenvectors']))
    else:
        pca = None

    return neural_network, pca


def save_session(nn: NeuralNetwork, unique_labels: np.ndarray = None, pca: PCA = None, f1_score=0, folder=None):
    """
    Saves parameters of a training session of a neural network to a file.
    :param folder: folder to save file in
    :param f1_score: last f1 score on test data for name
    :param nn: the trained neural network instance
    :param unique_labels: the different possible output classes
    :param pca: the pca instance storing eigenvectors, and number of principal components
    :return:
    """
    json_weights = convert_np_to_json(nn.weights)
    json_eigenvectors = convert_np_to_json(pca.eigenvectors) if pca is not None else None
    if not isinstance(unique_labels, list) and unique_labels is not None:
        unique_labels = unique_labels.tolist()
    timestamp_string = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    if folder is None:
        folder = 'saved_configs'
    fname = f'{folder}/config_{round(f1_score * 100)}_f1_{timestamp_string}.pkl'
    scaler_mean = nn.scaler.mean if nn.scaler_name == 'standard' else None
    scaler_std = nn.scaler.std if nn.scaler_name == 'standard' else None
    scaler_min_value = nn.scaler.min_value if nn.scaler_name == 'normal' else None
    scaler_max_value = nn.scaler.max_value if nn.scaler_name == 'normal' else None
    config = {
        'shape': nn.shape,
        'regression': nn.regression,
        'activation_function': nn.activation_function.__name__,
        'unique_labels': unique_labels,
        'pca_n': pca.n if pca is not None else None,
        'optimizer': str(nn.optimizer),
        'weights': json_weights,
        'eigenvectors': json_eigenvectors,
        'scaler': nn.scaler_name,
        'standard_scaler_mean': scaler_mean,
        'standard_scaler_std': scaler_std,
        'normal_scaler_min': scaler_min_value,
        'normal_scaler_max': scaler_max_value,
    }

    with lzma.open(fname, "wb") as fout:
        pickle.dump(config, fout)

    print(f'Session Configuration stored in file {fname}')


def convert_np_to_json(np_array):
    json_weights = []
    for weight in np_array:
        json_weights.append(weight.tolist())
    return json_weights
