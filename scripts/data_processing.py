import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset

def load_data(training_data_pth, test_data_pth):
    training_data = pd.read_csv(training_data_pth, usecols=[1, 2, 3])
    test_data = pd.read_csv(test_data_pth, usecols=[1, 2, 3])
    return training_data, test_data

def encode(Data):
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    num_chars = len(char_to_int)
    max_len = 16
    if isinstance(Data, str):
        Data = [Data]
    X = np.zeros((len(Data), max_len, num_chars))
    for i, seq in enumerate(Data):
        if len(seq) > 16:
            seq = seq[:16]
        for j, char in enumerate(seq):
            X[i, j, char_to_int[char]] = 1
    return pd.DataFrame(X.reshape(len(Data), -1)[:, :64])

def prepare_datasets(training_data, test_data):
    device = torch.device("cpu")
    Data_train = encode([training_data['starting_seq'].get(i)+training_data['binding'].get(i) for i in range(len(training_data))])
    labels_train = pd.DataFrame(training_data['fitness'])
    
    fitness_for_test=pd.DataFrame(test_data['fitness'])
    data_test=torch.tensor(encode(test_data['starting_seq']+test_data['binding']).values).float()
    data_test=data_test.to(device)

    labels_test=torch.tensor(fitness_for_test.values).float()
    labels_test=labels_test.view(labels_test.shape[0],1)
    labels_test=labels_test.to(device)
    return Data_train, labels_train, data_test, labels_test