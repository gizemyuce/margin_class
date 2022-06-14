from concurrent.futures import wait
import torch 

def min_l2_average_cf(z1s, z2s):
    z_seq = torch.cat((z1s, z2s), dim=0)
    z_mean = torch.mean(z_seq, dim=0)

    w = (z_mean/torch.norm(z_mean))

    return w

def min_l1_average_cf(z1s, z2s):
    d = z1s.size(dim=1)
    z_seq = torch.cat((z1s, z2s), dim=0)
    z_mean = torch.mean(z_seq, dim=0)

    w = (z_mean/torch.norm(z_mean))
    
    max_ind = torch.argmax(torch.abs(w))
    w_l1 = torch.zeros(d)
    if w[max_ind]>0:
        w_l1[max_ind] = 1
    else:
        w_l1[max_ind] = -1
    w = w_l1

    return w