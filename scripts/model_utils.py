import torch
import torch.nn as nn
import torch.optim as optim
from data_processing import encode


class Model(nn.Module):
    def __init__(self, input_shape=64, hidden_dim_1=32, hidden_dim_2=16, output_dim=1):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, output_dim),
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data = torch.randn(layer.weight.shape)
                layer.bias.data = torch.randn(layer.bias.shape)

    def forward(self, x):
        return self.layers(x)

    def regularization(self, weight, p_params, t_params):
        return weight * torch.mean(torch.square(p_params - t_params))

def train_model(x_test, y_test, train_data, labels_train_data, lr, epochs, trainloader, mean=0, std=1,):
    record = {"train_loss":[],"test_loss":[]}
    device = torch.device("cpu")
    torch.manual_seed(40)
    t = Model()                          # Initialize the trainable and the prior
    t.mean = mean
    t.std = std
    t.to(device)
    p = Model()
    p.mean = mean
    p.std = std
    p.to(device)

    Data_train = torch.tensor(train_data.values).float()
    Data_train=Data_train.to(device)
    labels_train = torch.tensor(labels_train_data.values).float()
    labels_train=labels_train.to(device)
    labels_train = labels_train.view(labels_train.shape[0],1)
    # Extract Prior's parameters
    p_params=[]
    for parameter in p.parameters():
        p_params.append(parameter.view(-1))
    p_params=torch.cat(p_params)

    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(t.parameters(), lr=lr)

    # Run the training loop
    for epoch in range(1, epochs+1):
        t.train()
        for i, data in enumerate(trainloader, 0):                             # Iterate over the DataLoader for training data

            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = t(inputs)
            loss = loss_function(outputs, targets)

            weight = 1.0/std
            t_params = []
            for parameter in t.parameters():
                t_params.append(parameter.view(-1))
            t_params=torch.cat(t_params)
            reg = t.regularization(weight, p_params, t_params)
            loss += reg
            loss.backward()
            optimizer.step()

        train_loss = loss_function(t(Data_train),   labels_train)
        train_loss = train_loss.detach().cpu().numpy()
        record['train_loss'].append(train_loss)

        t.eval()

        test_loss = loss_function(t(x_test), y_test)
        test_loss = test_loss.detach().cpu().numpy()
        record['test_loss'].append(test_loss)

        #print(f"[epoch {epoch}] Train:{train_loss:.5f} / Test loss:{test_loss:.5f}.")

    return t, record