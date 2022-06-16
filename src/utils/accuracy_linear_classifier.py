import torch

def test_error(w, x_test, y_test):
  w = w / torch.norm(w)
  pred = torch.sign(x_test @ w)
  err = (pred.int() != y_test.int()).sum()/float(y_test.size(0))*100
  return err