import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("muted")
import matplotlib.pyplot as plt

def create_data_sparse(p,n1,n2,n_test, s=1, random_flip_prob=0, seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # generating ground truth 
    w_gt = torch.zeros(p)
    w_gt[0:s] = 1/(s ** 0.5)

    # uniformly sampling points and labels
    xs = torch.randn((int(2*(n1+n2)), int(p)))
    ys_noiseless = torch.sign(xs @ w_gt)
    # xs = xs.numpy()
    # ys_noiseless = ys_noiseless.numpy()

    # generating the imbalance in the training data
    samples_one = xs[ys_noiseless==1]
    samples_one = samples_one[0:n1,:]

    samples_two = xs[ys_noiseless==-1]
    samples_two = samples_two[0:n2,:]

    if p==2:
        plt.figure()
        plt.scatter(samples_one[:,0], samples_one[:,1], color='blue')
        plt.scatter(samples_two[:,0], samples_two[:,1], color='red')
        plt.savefig('training_distribution.pdf')

    class_one = torch.Tensor(samples_one)
    class_two = torch.Tensor(samples_two)
    x_seq = torch.cat((class_one, class_two), dim=0)
    y_seq = torch.cat(
        (torch.ones(class_one.shape[0], dtype=torch.long), -torch.ones(class_two.shape[0], dtype=torch.long))
    )

    #add noise to the labels
    if random_flip_prob != 0:
        noise_mask = torch.bernoulli(random_flip_prob*torch.ones_like(y_seq))
        flip_to_0 = torch.logical_and(noise_mask==1, y_seq==1)
        flip_to_1 = torch.logical_and(noise_mask==1, y_seq==0)
        y_seq[flip_to_0] = 0
        y_seq[flip_to_1] = 1

        class_one = x_seq[y_seq==1]
        class_two = x_seq[y_seq==0]

        if p==2:
            plt.figure()
            plt.scatter(class_one[:,0], class_one[:,1], color='blue')
            plt.scatter(class_two[:,0], class_two[:,1], color='red')
            plt.savefig('training_distribution_with_noise.pdf')


    # genrating the test data without imbalanca and label noise
    xs_test = torch.randn((int(2*n_test), int(p)))
    ys_noiseless_test = torch.sign(xs_test @ w_gt)
    #ys_noiseless_test[ys_noiseless_test==-1] = 0
    
    return x_seq, y_seq, xs_test, ys_noiseless_test, class_one, -class_two