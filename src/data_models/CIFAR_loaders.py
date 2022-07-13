import torch
import torch.nn as nn
import numpy as np
import random 

import torchvision.datasets as datasets
from torchvision import transforms


import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import numpy as np


def get_binary_cifar10_loaders_cat_dog_3channels(n_train, n_val=2000, batch_size_train=None, batch_size=128, seed=0):

  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  
  if batch_size_train == None:
    batch_size_train = n_train

  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])

  
  train_dataset = datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
  
  val_dataset = datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)

  test_dataset = datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)


  classes = ('plane', 'car', 'bird', 'cat',
          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
  train_dataset = extract_classes_from_dataset(train_dataset, 3, 5)
  val_dataset = extract_classes_from_dataset(val_dataset, 3, 5)
  test_dataset = extract_classes_from_dataset(test_dataset, 3, 5)

  train_dataset, val_dataset = train_val_split(train_dataset, val_dataset, n_train, n_val)

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

def get_binary_cifar10_loaders_bird_plane(n_train, n_val=2000, batch_size_train=None, batch_size=128, seed=0):

  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  
  if batch_size_train == None:
    batch_size_train = n_train

  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])

  
  train_dataset = datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
  
  val_dataset = datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)

  test_dataset = datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)


  classes = ('plane', 'car', 'bird', 'cat',
          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
  train_dataset = extract_classes_from_dataset(train_dataset, 0, 2)
  val_dataset = extract_classes_from_dataset(val_dataset, 0, 2)
  test_dataset = extract_classes_from_dataset(test_dataset, 0, 2)

  train_dataset, val_dataset = train_val_split(train_dataset, val_dataset, n_train, n_val)

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

def extract_classes_from_dataset(dataset, class_1, class_2):

  idx = (np.array(dataset.targets)== int(class_1)) | (np.array(dataset.targets)== int(class_2))
  idx = np.argwhere(idx)
  idx = list(idx[:, 0])

  dataset.data = [dataset.data[i] for i in idx]
  dataset.targets =[0 if dataset.targets[i]==int(class_1) else 1 for i in idx]

  return dataset

def train_val_split(train_dataset, val_dataset, n_train, n_val):

  if n_val > len(train_dataset) - n_train:
      n_val = len(train_dataset) - n_train

  index_sub = np.random.choice(np.arange(len(train_dataset)), int(n_train+n_val), replace=False)
  ind_train = index_sub[:n_train]
  ind_val = index_sub[n_train:-1]

  train_dataset.data = [train_dataset.data[i] for i in ind_train]
  train_dataset.targets = [train_dataset.targets[i] for i in ind_train]

  val_dataset.data = [val_dataset.data[i] for i in ind_val]
  val_dataset.targets = [val_dataset.targets[i] for i in ind_val]

  return train_dataset, val_dataset

#   # replacing attribute
#   train_dataset.data = train_dataset.data[ind_train]
#   train_dataset.targets = train_dataset.targets[ind_train]

#   val_dataset.data = val_dataset.data[ind_val]
#   val_dataset.targets = val_dataset.targets[ind_val]


# Transformations
RC = transforms.RandomCrop(32, padding=4)
RHF = transforms.RandomHorizontalFlip()
RVF = transforms.RandomVerticalFlip()
NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
TT = transforms.ToTensor()
TPIL = transforms.ToPILImage()

# Transforms object for trainset with augmentation
transform_with_aug = transforms.Compose([TPIL, RC, RHF, TT, NRM])
# Transforms object for testset with NO augmentation
transform_no_aug = transforms.Compose([TT, NRM])

# Downloading/Louding CIFAR10 data
# , transform = transform_with_aug)
trainset = CIFAR10(root='./data', train=True, download=True)
# , transform = transform_no_aug)
testset = CIFAR10(root='./data', train=False, download=True)
classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
             'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

# Separating trainset/testset data/label
x_train = trainset.data
x_test = testset.data
y_train = trainset.targets
y_test = testset.targets

# Define a function to separate CIFAR classes by class index


def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i


class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc=transform_no_aug):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class

# ================== Usage ================== #


# Let's choose cats (class 3 of CIFAR) and dogs (class 5 of CIFAR) as trainset/testset
