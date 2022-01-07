from comet_ml import Experiment

import os
os.system("pip install -r ../../requirements.txt")

import itertools
import torch
from torch import nn
from torch.optim import SGD
import w1d4_tests
import numpy as np
import matplotlib.pyplot as plt
import gin
import time

fname = 'small_moon.jpg'
data_train, data_test =  w1d4_tests.load_image(fname)

class OurNet(nn.Module):
    def __init__(self, P, H, K):
        super(OurNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(P, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, K)
        )
    
    def forward(self, x):
        return self.layers(x)
    
@gin.configurable
def make_grid(possible_values):
    a = possible_values.values()
    pvs = list(itertools.product(*a))

    ld = []
    for pv in pvs:
        dict = {}
        for e, key in enumerate(possible_values.keys()):
            dict[key] = pv[e]
        ld.append(dict)

    return ld

gin.enter_interactive_mode()
@gin.configurable
def train(model, dataloader, lr, momentum):

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum)
    # optimizer = RMSPROP(model.parameters(), 0.0002, 0.001, 0.001, 0, 0.7)
    
    loss_function = nn.L1Loss()

    model.train()

    for i, data in enumerate(dataloader):
        inputs, target = data

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_function(outputs, target)

        loss.backward()
        
        optimizer.step()

    return model

from torch import mean
import torch as t

def evaluate(model, dataloader):
    model.eval()
    loss_function = nn.L1Loss()

    losses = []

    for i, data in enumerate(dataloader):
        inputs, target = data


        outputs = model(inputs)

        loss = loss_function(outputs, target)

        losses.append(loss)

    return float(mean(t.tensor(losses)))

@gin.configurable
class SGDHyperparameters:
    def __init__(self, lr, momentum):
        self.lr = lr
        self.momentum = momentum

        self.hyperparameter_dict = {
            'lr': self.lr,
            'momentum': self.momentum
        }

@gin.configurable
def sgd_hyperparameter_search(dict):
    l = make_grid(dict)
    gin.enter_interactive_mode()

    current_time = time.localtime()
    current_time = time.strftime("%H_%M_%S", current_time)

    with gin.unlock_config():
        gin.parse_config_file(config_file="config.gin")

        for d in l:
            experiment = Experiment(
                api_key="xs16WsBDV0OjJyQ9XWoTyLJnU",
                project_name=f"sgd_mlab_{current_time}",
                workspace="dnlmy",
            )

            experiment.log_parameter('lr', d['lr'])
            experiment.log_parameter('momentum', d['momentum'])

            on = OurNet(2,400,3)
            epochs = 25

            train_losses = []
            test_losses = []            

            for epoch in range(epochs):

                experiment.set_epoch(epoch)

                print("epoch: ", epoch)

                train_loss = evaluate(on, data_train)
                test_loss = evaluate(on, data_test)

                experiment.log_metric('train_loss', train_loss)
                experiment.log_metric('test_loss', test_loss)

                train_losses.append(train_loss)
                test_losses.append(test_loss)

                on = train(on, data_test, lr = d["lr"], momentum = d["momentum"])

            experiment.end()

sgd_hyperparameter_search(gin.REQUIRED)