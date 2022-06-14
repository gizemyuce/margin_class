import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def create_data(class_one_train_num=20, class_two_train_num=200, class_one_test_num=30, class_two_test_num=30,  n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, class_sep=1.0):
  num_train_samples = class_one_train_num + class_two_train_num
  num_test_samples = class_one_test_num + class_two_test_num
  num_samples = num_train_samples + num_test_samples
  X, y = make_classification(n_samples=num_samples ,  n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, class_sep=class_sep)

  samples_one = X[y==0]
  labels_one = y[y==0]

  X_one_train, X_one_test, y_one_train, y_one_test = train_test_split(samples_one, labels_one, test_size=samples_one.shape[0]-class_one_train_num, random_state=42)

  samples_two = X[y==1]
  labels_two = y[y==1]

  X_two_train, X_two_test, y_two_train, y_two_test = train_test_split(samples_two, labels_two, test_size=samples_two.shape[0]-class_two_train_num, random_state=42)

  class_one = torch.Tensor(X_one_train)
  class_two = torch.Tensor(X_two_train)
  x_seq = torch.cat((class_one, class_two), dim=0)
  y_seq = torch.cat(
      (torch.ones(class_one.shape[0], dtype=torch.long), -torch.ones(class_two.shape[0], dtype=torch.long))
  )
  
  dataset_train = torch.utils.data.TensorDataset(x_seq, y_seq)
  dataloader_train = torch.utils.data.DataLoader(
      dataset=dataset_train, batch_size=num_train_samples, shuffle=True
  )

  class_one_test = torch.Tensor(X_one_test)
  class_two_test = torch.Tensor(X_two_test)
  x_seq_test = torch.cat((class_one_test, class_two_test), dim=0)
  y_seq_test = torch.cat(
      (torch.ones(class_one_test.shape[0], dtype=torch.long), -torch.ones(class_two_test.shape[0], dtype=torch.long))
  )
  
  dataset_test = torch.utils.data.TensorDataset(x_seq_test, y_seq_test)
  dataloader_test = torch.utils.data.DataLoader(
      dataset=dataset_test, batch_size=num_test_samples, shuffle=True
  )  

  return x_seq, y_seq, x_seq_test, y_seq_test, class_one, -class_two