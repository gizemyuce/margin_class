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
from src.data_models.CIFAR_loaders import get_binary_cifar10_loaders_bird_plane, get_binary_cifar10_loaders_cat_dog_3channels, get_cifar10_loaders
from src.data_models.FMnist_loaders import get_fmnist_loaders_3channels
from src.utils.loss_functions import *
from src.architectures.CNN import CNNModel
from src.architectures.Convnet import ConvNet
from src.architectures.Resnet import ResNetMulti, ResNet_50_Multi

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


hyperparameter_defaults = dict(
    learning_rate = 0.07343,
    epochs = 1000,
    n=256,
    loss_type='ce',
    dataset = 'FMNIST',
    architecture = 'ResNet',
    seed = 0,
    momentum=0.7733,
    weight_decay=0,
    test=True,
    left_loss='exp',
    avg_mrgn_loss_type = '-',
    alpha=1.0,
    beta=1.0,
    scheduler_step=500,
    scheduler_gamma = 0.9,
    batchsize_train = None,
    combo_avg=0.5, 
    train_only_linear = False,
    )


wandb.init(config=hyperparameter_defaults, project="avg_margin_fmnist_multiclass", entity='sml-eth')
config = wandb.config

torch.manual_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)

def main():

  torch.manual_seed(config.seed)
  random.seed(config.seed)
  np.random.seed(config.seed)
  
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
        

  if config.dataset == 'FMNIST':
    train_loader, val_loader, test_loader = get_fmnist_loaders_3channels(config.n, batch_size_train=config.batchsize_train, batch_size=128, seed=config.seed)
  elif config.dataset == 'CIFAR10':
    train_loader, val_loader, test_loader = get_cifar10_loaders(config.n, batch_size_train=config.batchsize_train, batch_size=128, seed=config.seed)

  if config.loss_type == 'ce':
    criterion = nn.CrossEntropyLoss(reduction="none")
  elif config.loss_type == 'poly':
    criterion = PolynomialLoss(type="logit", alpha=config.alpha, beta=config.beta)
  elif config.loss_type == 'poly-max':
    criterion = MCPolynomialLoss_max(type="logit", alpha=config.alpha, beta=config.beta)
  elif config.loss_type == 'poly-sum':
    criterion = MCPolynomialLoss_sum(type="logit", alpha=config.alpha, beta=config.beta)
  elif config.loss_type == 'avg-sum':
    criterion = AverageMarginlLoss_sum(type = config.avg_mrgn_loss_type)
  elif config.loss_type == 'avg-max':
    criterion = AverageMarginlLoss_max(type = config.avg_mrgn_loss_type)
  elif config.loss_type == 'avg-max-zero':
    criterion = AverageMarginlLoss_max_zero(type = config.avg_mrgn_loss_type)
  elif config.loss_type == 'multimargin':
    criterion = nn.MultiMarginLoss()
  elif config.loss_type == 'avg-max-hinge':
    criterion = AverageMarginlLoss_max_hinge()
  elif config.loss_type == 'combo':
    criterion_1 = AverageMarginlLoss_max_hinge()
    criterion_2 = nn.CrossEntropyLoss(reduction="none")


  if config.architecture == 'CNN':
    model = CNNModel()
  elif config.architecture == 'Convnet':
    model = ConvNet()
  elif config.architecture == 'ResNet':
    model = ResNetMulti()
  elif config.architecture == 'ResNet50':
    model = ResNet_50_Multi()

  model = torch.load("resnet18_ce_256.pb")

    
  if config.train_only_linear:
    frozen_layers = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1,model.layer2,model.layer3,model.layer4, model.avgpool ]
    for layer in frozen_layers:
      #layer.trainable = False
      for param in layer.parameters():
        param.requires_grad = False

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

        if epoch==0 and i == 0:
          print(labels)

        # if torch.cuda.is_available():
        #   imgs = imgs.cuda()
        #   labels = labels.cuda()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        if config.loss_type=='combo':
          loss = torch.mean(config.combo_avg*criterion_1(outputs, labels)+(1-config.combo_avg)*criterion_2(outputs, labels))
        else:
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

    conf_train = wandb.plot.confusion_matrix(probs=None,
                        y_true = labels.cpu().detach().numpy(), preds=torch.argmax(outputs, dim=1).cpu().detach().numpy(),
                        class_names=label_names)


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

        for label in range(10):
            correct_arr[label] += (((predicted == labels) & (labels==label)).sum())
            total_arr[label] += (labels == label).sum()

    accuracy = correct / total

    metrics = {'accuracy': accuracy, 'loss': loss, 'train_accuracy': train_accuracy}
    
    for label in range(10):
        metrics['Accuracy ' + label_names[label]] = correct_arr[label] / total_arr[label]

    if config.loss_type == 'poly':
      metrics['margin_ratio'] = margin_ratio

    metrics["conf_mat_train"] = conf_train

    metrics["conf_mat_test"] = wandb.plot.confusion_matrix(probs=None,
                        y_true = labels.cpu().detach().numpy(), preds=predicted.cpu().detach().numpy(),
                        class_names=label_names)

    wandb.log(metrics)

    # Print Loss
    print('Epoch: {0} Loss: {1:.4f} Train_Acc:{3: .4f} Val_Accuracy: {2:.4f}'.format(epoch, loss, accuracy, train_accuracy))

    torch.save(model, "resnet18_ce_256.pb")


  #torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
  wandb.finish()


if __name__ == '__main__':
   main()
