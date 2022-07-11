import torch
import torch.nn as nn
import numpy as np
import random 

import torchvision.datasets as datasets
from torchvision import transforms



def get_binary_cifar10_loaders_cat_dog_3channels(n_train, n_val=2000, batch_size_train=None, batch_size=128, seed=0):

  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  
  if batch_size_train == None:
    batch_size_train = n_train

  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  train_dataset = datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
  val_dataset = datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)

  test_dataset = datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)


  classes = ('plane', 'car', 'bird', 'cat',
          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  
  idx = (train_dataset.targets==3) | (train_dataset.targets==5)
  train_dataset.targets = train_dataset.targets[idx]
  train_dataset.data = train_dataset.data[idx]

  idx = (val_dataset.targets==3) | (val_dataset.targets==5)
  val_dataset.targets = val_dataset.targets[idx]
  val_dataset.data = val_dataset.data[idx]

  idx_test = (test_dataset.targets==3) | (test_dataset.targets==5)
  test_dataset.targets = test_dataset.targets[idx_test]
  test_dataset.data = test_dataset.data[idx_test]

  train_dataset.targets[train_dataset.targets==3] = 0
  train_dataset.targets[train_dataset.targets==5] = 1

  val_dataset.targets[val_dataset.targets==3] = 0
  val_dataset.targets[val_dataset.targets==5] = 1

  test_dataset.targets[test_dataset.targets==3] = 0
  test_dataset.targets[test_dataset.targets==5] = 1

  # subset training set
  index_sub = np.random.choice(np.arange(len(train_dataset)), int(n_train+n_val), replace=False)
  ind_train = index_sub[:n_train]
  ind_val = index_sub[n_train:-1]

  # replacing attribute
  train_dataset.data = train_dataset.data[ind_train]
  train_dataset.targets = train_dataset.targets[ind_train]

  val_dataset.data = val_dataset.data[ind_val]
  val_dataset.targets = val_dataset.targets[ind_val]


  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size_train,
                                              shuffle=True)
  
  val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

  test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

  return train_loader, val_loader, test_loader  
