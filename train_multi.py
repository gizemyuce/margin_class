from platform import architecture
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms
import torch.optim as optim
import time
import tqdm as tqdm
from torch.autograd import Variable

import random

import wandb
import sys

import seaborn as sns
sns.set_palette("muted")
import math

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Linear, Sequential

import wandb
import os
import numpy as np

from src.utils.loss_functions import *
from src.architectures.CNN import CNNModel
from src.architectures.Convnet import ConvNet
from src.architectures.Resnet import ResNetMulti
from src.data_models.FMnist_loaders import get_fmnist_loaders_3channels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hyperparameter_defaults = dict(
    learning_rate = 0.1,
    epochs = 1000,
    n=256,
    loss_type= 'poly-max',                #'ce',
    dataset = 'FashionMNIST',
    architecture = 'ResNet',
    seed = 0,
    momentum=0.9,
    weight_decay=0,
    beta = 1000.,
    alpha=1.,
    )

wandb.init(config=hyperparameter_defaults, project="fmnist_multi")
config = wandb.config

torch.manual_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)

def main():
  alpha=1
  
  label_names = [
        "T-shirt or top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Boot"]

  train_loader, val_loader, test_loader = get_fmnist_loaders_3channels(config.n, batch_size_train=None, batch_size=128, seed=config.seed)

  if config.loss_type == 'ce':
    criterion = nn.CrossEntropyLoss(reduction="none")
  elif config.loss_type == 'poly':
    criterion = PolynomialLoss(type="logit", alpha=config.alpha, beta=config.beta)
  elif config.loss_type == 'poly-max':
    criterion = MCPolynomialLoss_max(type="logit", alpha=config.alpha, beta=config.beta)
  elif config.loss_type == 'poly-sum':
    criterion = MCPolynomialLoss_sum(type="logit", alpha=config.alpha, beta=config.beta)

  if config.architecture == 'CNN':
    model = CNNModel()
  elif config.architecture == 'Convnet':
    model = ConvNet()
  elif config.architecture == 'ResNet':
    model = ResNetMulti()

  
    
  model = model.to(device) 

  wandb.watch(model)
  
  optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
  
  #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, last_epoch=- 1, verbose=False)

  iter=0
  for epoch in range(config['epochs']):
    train_acc=[]
    model.train()

    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images)
        labels = Variable(labels)

        images=images.to(device)
        labels=labels.to(device)

        # if torch.cuda.is_available():
        #   imgs = imgs.cuda()
        #   labels = labels.cuda()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = torch.mean(criterion(outputs, labels))

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()
        

        train_acc.append((torch.sum((torch.argmax(outputs, dim=1) == labels))).float())

        iter += 1

    #scheduler.step()
    train_accuracy = sum(train_acc)/config.n

    # Calculate Val Accuracy
    # model.eval()

    correct = 0.0
    correct_arr = [0.0] * 10
    total = 0.0
    total_arr = [0.0] * 10

    # Iterate through test dataset
    for images, labels in val_loader:
        images = Variable(images)
        images= images.to(device)
        labels = Variable(labels).to(device)

        # Forward pass only to get logits/output
        outputs = model(images)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)
        correct += (predicted == labels).sum()

        for label in range(10):
            correct_arr[label] += (((predicted == labels) & (labels==label)).sum())
            total_arr[label] += (labels == label).sum()

    accuracy = correct / total

    metrics = {'accuracy': accuracy, 'loss': loss, 'train_accuracy': train_accuracy}
    for label in range(10):
        metrics['Accuracy ' + label_names[label]] = correct_arr[label] / total_arr[label]

    wandb.log(metrics)

    # Print Loss
    print('Epoch: {0} Loss: {1:.4f} Train_Acc:{3: .4f} Val_Accuracy: {2:.4f}'.format(epoch, loss, accuracy, train_accuracy))


  torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
  wandb.finish()


if __name__ == '__main__':
   main()
