from sklearn.metrics import average_precision_score
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
from torch.optim.lr_scheduler import StepLR

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

from src.data_models.FMnist_loaders import get_binary_fmnist_loaders_01, get_binary_fmnist_loaders_24, get_binary_fmnist_loaders_24_3channels
from src.data_models.CIFAR_loaders import get_binary_cifar10_loaders_bird_plane, get_binary_cifar10_loaders_cat_dog_3channels
from src.utils.loss_functions import AverageMarginlLoss, PolynomialLoss, PolynomialLoss_pure
from src.architectures.Resnet import ResNetBinary
from src.architectures.Convnet import ConvNet_binary
from src.architectures.CNN import CNNModel_binary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


hyperparameter_defaults = dict(
    learning_rate = 0.025,
    epochs = 1000,
    n=8000,
    loss_type='ce',
    dataset = 'CIFAR10-cd',
    architecture = 'ResNet',
    seed = 0,
    momentum=0.9,
    weight_decay=0,
    test=False,
    left_loss='exp',
    avg_mrgn_loss_type = '-',
    alpha=1.0,
    beta=1.0,
    scheduler_step=300,
    scheduler_gamma = 0.9,
    batchsize_train = 128,
    )


wandb.init(config=hyperparameter_defaults, project="avg_margin_cifar10", entity='sml-eth')
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
        
  label_names = [
        "Class1",
        "Class2"]

  if config.dataset == 'FashionMNIST-binary01':
    train_loader, val_loader, test_loader = get_binary_fmnist_loaders_01(config.n, batch_size_train=config.batchsize_train, batch_size=128, seed=config.seed)
  elif config.dataset == 'FashionMNIST-binary24':
    if config.architecture == 'ResNet':
      train_loader, val_loader, test_loader = get_binary_fmnist_loaders_24_3channels(config.n, batch_size_train=config.batchsize_train, batch_size=128, seed=config.seed)
    else:
      train_loader, val_loader, test_loader = get_binary_fmnist_loaders_24(config.n, batch_size_train=config.batchsize_train, batch_size=128, seed=config.seed)
  elif config.dataset == 'CIFAR10-cd':
    train_loader, val_loader, test_loader = get_binary_cifar10_loaders_cat_dog_3channels(config.n, batch_size_train=config.batchsize_train, batch_size=128, seed=config.seed)
  elif config.dataset == 'CIFAR10-bp':
    train_loader, val_loader, test_loader = get_binary_cifar10_loaders_bird_plane(config.n, batch_size_train=config.batchsize_train, batch_size=128, seed=config.seed)

  if config.loss_type == 'ce':
    criterion = nn.CrossEntropyLoss(reduction="none")
  elif config.loss_type == 'poly':
    criterion = PolynomialLoss(type=config.left_loss, alpha=config.alpha, beta=config.beta)
  elif config.loss_type == 'poly-pure':
    criterion = PolynomialLoss_pure(type=config.left_loss, alpha=config.alpha, beta=config.beta)
  elif config.loss_type == 'avg':
    criterion = AverageMarginlLoss(type = config.avg_mrgn_loss_type)

  if config.architecture == 'CNN':
    model = CNNModel_binary()
  elif config.architecture == 'Convnet':
    model = ConvNet_binary()
  elif config.architecture == 'ResNet':
    model = ResNetBinary()
    
  model = model.to(device) 

  # wandb.watch(model)
  
  optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
  scheduler = StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)
  
  iter=0
  for epoch in range(config['epochs']):
    train_acc=[]
    margin_sum = []
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

        if config.loss_type == 'poly':
          target_sign = 2 * labels - 1
          margin_scores = (outputs[:, 1] - outputs[:, 0]) * target_sign
          indicator = margin_scores <= config.beta
          margin_sum.append(torch.sum(indicator))

        iter += 1
    
    scheduler.step()

    train_accuracy = sum(train_acc)/config.n
    margin_ratio = sum(margin_sum)/config.n

    # Calculate Val Accuracy
    # model.eval()

    correct = 0.0
    correct_arr = [0.0] * 10
    total = 0.0
    total_arr = [0.0] * 10

    if config.test:
      loader=test_loader
    else:
      loader=val_loader

    # Iterate through test dataset
    for images, labels in loader:
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

        for label in range(2):
            correct_arr[label] += (((predicted == labels) & (labels==label)).sum())
            total_arr[label] += (labels == label).sum()

    accuracy = correct / total

    metrics = {'accuracy': accuracy, 'loss': loss, 'train_accuracy': train_accuracy}
    
    for label in range(2):
        metrics['Accuracy ' + label_names[label]] = correct_arr[label] / total_arr[label]

    if config.loss_type == 'poly':
      metrics['margin_ratio'] = margin_ratio

    wandb.log(metrics)

    # Print Loss
    print('Epoch: {0} Loss: {1:.4f} Train_Acc:{3: .4f} Val_Accuracy: {2:.4f}'.format(epoch, loss, accuracy, train_accuracy))

  if config.loss_type=='ce':
    ce_test_final = accuracy
  elif config.loss_type=='poly':
    poly_test_final = accuracy

  #torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
  wandb.finish()


if __name__ == '__main__':
   main()
