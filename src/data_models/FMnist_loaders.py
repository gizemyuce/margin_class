import torch
import torch.nn as nn
import numpy as np
import random 

import torchvision.datasets as datasets
from torchvision import transforms



def get_fmnist_loaders_3channels(n_train, n_val=10000, batch_size_train=None, batch_size=128, seed=torch.seed):

  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  
  if batch_size_train == None:
    batch_size_train = n_train

  normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,)),
                                  # expand chennel from 1 to 3 to fit 
                                  # ResNet pretrained model
                                  transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                  ])

  train_dataset = datasets.FashionMNIST(root='./data',
                              train=True,
                              transform=transform,
                              download=True
                              )
  
  val_dataset = datasets.FashionMNIST(root='./data',
                              train=True,
                              transform=transform,
                              download=True
                              )

  test_dataset = datasets.FashionMNIST(root='./data',
                              train=False,
                              transform=transform,
                              )

  # subset training set
  index_sub = np.random.choice(np.arange(len(train_dataset)), int(n_train+n_val), replace=False)
  ind_train = index_sub[:n_train]
  ind_val = index_sub[n_train:-1]

  # replacing attribute
  train_dataset.data = train_dataset.data[ind_train]
  train_dataset.targets = train_dataset.targets[ind_train]

  val_dataset.data = val_dataset.data[ind_val]
  val_dataset.targets = val_dataset.targets[ind_val]

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

def get_binary_fmnist_loaders_01(n_train, n_val=2000, batch_size_train=None, batch_size=128):
  
  if batch_size_train == None:
    batch_size_train = n_train

  normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])

  train_dataset = datasets.FashionMNIST(root='./data',
                              train=True,
                              transform=transform,
                              download=True
                              )
  
  val_dataset = datasets.FashionMNIST(root='./data',
                              train=True,
                              transform=transform,
                              download=True
                              )

  test_dataset = datasets.FashionMNIST(root='./data',
                              train=False,
                              transform=transform,
                              )
  
  idx = (train_dataset.targets==0) | (train_dataset.targets==1)
  train_dataset.targets = train_dataset.targets[idx]
  train_dataset.data = train_dataset.data[idx]

  idx = (val_dataset.targets==0) | (val_dataset.targets==1)
  val_dataset.targets = val_dataset.targets[idx]
  val_dataset.data = val_dataset.data[idx]

  idx_test = (test_dataset.targets==0) | (test_dataset.targets==1)
  test_dataset.targets = test_dataset.targets[idx_test]
  test_dataset.data = test_dataset.data[idx_test]
  
  # subset training set
  index_sub = np.random.choice(np.arange(len(train_dataset)), int(n_train+n_val), replace=False)
  ind_train = index_sub[:n_train]
  ind_val = index_sub[n_train:-1]

  # replacing attribute
  train_dataset.data = train_dataset.data[ind_train]
  train_dataset.targets = train_dataset.targets[ind_train]

  val_dataset.data = val_dataset.data[ind_val]
  val_dataset.targets = val_dataset.targets[ind_val]

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

def get_binary_fmnist_loaders_24(n_train, n_val=2000, batch_size_train=None, batch_size=128):
  
  if batch_size_train == None:
    batch_size_train = n_train

  normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])

  train_dataset = datasets.FashionMNIST(root='./data',
                              train=True,
                              transform=transform,
                              download=True
                              )
  
  val_dataset = datasets.FashionMNIST(root='./data',
                              train=True,
                              transform=transform,
                              download=True
                              )

  test_dataset = datasets.FashionMNIST(root='./data',
                              train=False,
                              transform=transform,
                              )
  
  
  idx = (train_dataset.targets==2) | (train_dataset.targets==4)
  train_dataset.targets = train_dataset.targets[idx]
  train_dataset.data = train_dataset.data[idx]

  idx = (val_dataset.targets==2) | (val_dataset.targets==4)
  val_dataset.targets = val_dataset.targets[idx]
  val_dataset.data = val_dataset.data[idx]

  idx_test = (test_dataset.targets==2) | (test_dataset.targets==4)
  test_dataset.targets = test_dataset.targets[idx_test]
  test_dataset.data = test_dataset.data[idx_test]

  train_dataset.targets[train_dataset.targets==2] = 0
  train_dataset.targets[train_dataset.targets==4] = 1

  val_dataset.targets[val_dataset.targets==2] = 0
  val_dataset.targets[val_dataset.targets==4] = 1

  test_dataset.targets[test_dataset.targets==2] = 0
  test_dataset.targets[test_dataset.targets==4] = 1

  # subset training set
  index_sub = np.random.choice(np.arange(len(train_dataset)), int(n_train+n_val), replace=False)
  ind_train = index_sub[:n_train]
  ind_val = index_sub[n_train:-1]

  # replacing attribute
  train_dataset.data = train_dataset.data[ind_train]
  train_dataset.targets = train_dataset.targets[ind_train]

  val_dataset.data = val_dataset.data[ind_val]
  val_dataset.targets = val_dataset.targets[ind_val]

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
