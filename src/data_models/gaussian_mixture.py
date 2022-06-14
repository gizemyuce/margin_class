import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("muted")
import matplotlib.pyplot as plt

def create_data_mixture(p,n1,n2,n_test, s=1, random_flip_prob=0, SNR=10, outlier=True, outlier_strenght=100):
    
    mu_1 = torch.zeros(p)
    mu_2 = torch.zeros(p)

    # mu_1[0:s] = 1/(s ** 0.5) * (p*SNR) **0.5
    # mu_2[0:s] = - 1/(s ** 0.5) * (p*SNR) ** 0.5

    mu_1[0:s] = SNR
    mu_2[0:s] = - SNR

    samples_one = torch.randn((n1, p)) + mu_1
    samples_two = torch.randn((n2, p)) + mu_2

    if outlier_strenght != 0:
        samples_one[1,:] += mu_1
        #samples_one[1,s] = outlier_strenght*(n1+n2)* 1/(s ** 0.5) * np.sqrt(p*SNR)
        samples_one[1,s] = outlier_strenght*(n1+n2)* SNR

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
    xs_test = torch.cat((torch.randn((int(n_test), p)) + mu_1, torch.randn((int(n_test), p)) + mu_2), dim=0)
    ys_noiseless_test = torch.cat(
        (torch.ones(int(n_test), dtype=torch.long), -torch.ones(int(n_test), dtype=torch.long))
    )
    
    return x_seq, y_seq, xs_test, ys_noiseless_test, class_one, -class_two