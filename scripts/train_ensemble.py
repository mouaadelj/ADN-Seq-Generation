from data_processing import load_data, prepare_datasets
from model_utils import Model, train_model
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import torch
import random
import os

current_dir = os.path.abspath(os.getcwd())
parent_dir = os.path.dirname(current_dir)


def train_ensemble(training_data_pth, test_data_pth, num_bootstraps=6, lr=0.01, batch_size=50, epochs=20, mean=0, std=5, models_dir = '../models'):
   
    models=[]
    records=[]    
    
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(models_dir, "model_1.pth")
    
    if os.path.exists(model_path):

        print(f"Models found at {models_dir}, loading ...")
        for i in range(num_bootstraps) : 
            path = '../models/model_'+ str(i+1) +'.pth'
            import_model = Model()
            import_model.load_state_dict(torch.load(path))
            models.append(import_model)
    else :    
        train, test = load_data(training_data_pth, test_data_pth)
        train_data, labels_train_data, x_test, y_test = prepare_datasets(train, test)
        loaders=bootstrap_generator(train_data, labels_train_data, num_bootstraps, batch_size)

        i=0
        for bootstrap_trainloader in loaders:
            i=i+1
            print(f"Training model {i} ")
            model,record = train_model(x_test, y_test, train_data, labels_train_data, lr, epochs, bootstrap_trainloader, mean, std)
            models.append(model)
            records.append(record)

    return models, records


def bootstrap_generator(train_data, labels_train_data, num_bootstraps, batch_size):

    loaders=[]
    for i in range(num_bootstraps):

        bootstrap_indices = [random.randint(0, train_data.shape[0] - 1) for _ in range(train_data.shape[0])]     # random sampling from dataset
        bootstrap_dataset = train_data.iloc[bootstrap_indices,:]
        bootstrap_dataset_l = labels_train_data.iloc[bootstrap_indices,:]
        bootstrap_x_train = torch.tensor(bootstrap_dataset.values).float()
        bootstrap_y_train = torch.tensor(bootstrap_dataset_l.values).float()
        bootstrap_y_train = bootstrap_y_train.view(bootstrap_y_train.shape[0],1)
        bootstrap_data_train = TensorDataset(bootstrap_x_train, bootstrap_y_train)
        bootstrap_trainloader = DataLoader(bootstrap_data_train, batch_size=batch_size, shuffle=True)
        loaders.append(bootstrap_trainloader)
    return loaders

def get_ensemble_predections(num_bootstraps, models,X):
    
    y_ensemble_predections=0#torch.zeros(X.shape[0],1).to(device)
    for model in models:
        model.eval()
        with torch.no_grad():
            y_ensemble_predections = y_ensemble_predections + model.forward(X)
    y_ensemble_predections = y_ensemble_predections/num_bootstraps
    y_ensemble_predections = y_ensemble_predections.view(y_ensemble_predections.shape[0],1)
    return y_ensemble_predections


if __name__ == "__main__":
    train_ensemble("data/training_data.csv", "data/test_data.csv")