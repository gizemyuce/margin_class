import numpy as np
from scipy.stats import norm
from sklearn import svm
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import pickle as pkl

import torch
import numpy as np
import torch.optim as optim
import matplotlib as mpl
import random 

import seaborn as sns
sns.set_palette("muted")
import matplotlib.pyplot as plt
import math
# %matplotlib inline

import wandb

import cvxpy as cp

from src.data_models.gaussian_marginal import create_data_sparse
from src.data_models.gaussian_mixture import create_data_mixture
from src.utils.accuracy_linear_classifier import test_error
from src.margin_estimators.closed_form import *
from src.margin_estimators.interpolating_const import *
from src.margin_estimators.svc import *

from src.margin_estimators import closed_form as cf
from src.margin_estimators import interpolating_const as interp

hyperparameter_defaults = dict(
    n_train= 100,
    n_test= int(1e4),
    n_features= 5000,
    tau = 1,
    SNR = 3,
    s= 1,
    label_noise_prob= 0,
    data_mixture = False,
    seed = 0,
    estimator_type = 'l2-avg-intp',
    )

wandb.init(config=hyperparameter_defaults, project="margin_classifiers")
config = wandb.config

torch.manual_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)


def main():

    n1 = min(int(np.round(config.tau * config.n_train/(1.+config.tau))), int(config.n_train)-1)
    n2 = int(config.n_train) - int(n1)
    n1, n2 = max(n1, n2), min(n1, n2)

    tau = n1/n2

    wandb.config.tau = tau
    config.tau = tau
 
    if config.data_mixture:
        xs, ys, x_seq_test, y_seq_test, z1s, z2s = create_data_mixture(config.n_features,n1,n2,config.n_test, s=config.s, random_flip_prob=config.label_noise_prob, SNR=config.SNR, outlier_strenght=config.outlier_strength, seed=config.seed)
    else:
        xs, ys, x_seq_test, y_seq_test, z1s, z2s = create_data_sparse(config.n_features,n1,n2,config.n_test, s=config.s, random_flip_prob=config.label_noise_prob, seed=config.seed)
        
        w_gt = torch.zeros(config.n_features)
        w_gt[0:config.s] = 1/(config.s ** 0.5)

    z_seq = torch.cat((z1s, z2s), dim=0)
    z_mean = torch.mean(z_seq, dim=0)
    
    a_vals = [0.]

    if config.estimator_type == 'l1-avg-intp':
        w = l1_average_interp(z1s, z2s, linear_program=True)
    elif config.estimator_type == 'l2-avg-intp':
        w = l2_average_interp(z1s, z2s)
    elif config.estimator_type == 'l2-min':
        w = l2_min_margin(xs, ys)
    elif config.estimator_type == 'l1-min':
        w = l1_min_margin(xs, ys)
    elif config.estimator_type == 'l1-avg-cf':
        w = min_l1_average_cf(z1s, z2s)
    elif config.estimator_type == 'l2-avg-cf':
        w = min_l2_average_cf(z1s, z2s)

    err_train = test_error(w, xs, ys)
    err = test_error(w, x_seq_test, y_seq_test)

    print("train_err={}, err={}".format(err_train, err))

    wandb.log({"err_train": err_train,
            "err_test": err,
            "estimator": w,
            "est_error": torch.norm(w - w_gt),
            "difference": w - w_gt,
            "aspect_ratio": config.n_features/config.n_train
            })


    # if l2:
    #     # l2 margin
    #     # clf = svm.LinearSVC(loss='hinge', fit_intercept=False, C=1e5, max_iter=int(1e8))
    #     # clf.fit(xs, ys)
    #     # wmm = -torch.Tensor(clf.coef_.flatten())
    #     # #print(wmm)
    #     # perf_train_mm = clf.score(xs, ys)
    #     # err_train_mm = 100*(1.-perf_train_mm)
    #     # err_mm = test_error(wmm, x_seq_test, y_seq_test)

    #     _,_, wmm = solve_svc_problem(xs, ys, p=2) 
    #     #print(wmm.value)
    #     wmm = torch.Tensor(wmm.value)
    #     err_train_mm = test_error(wmm, xs,ys)
    #     err_mm = test_error(wmm, x_seq_test, y_seq_test)

    #     print("CMM train_err={}, err={}".format( err_train_mm, err_mm))

    #     errs_avm_poly = []
    #     errs_train_avm_poly = []

    #     for a in a_vals:

    #         b = tau**a

    #         A_ub = torch.cat( [torch.cat([torch.eye(d), -torch.eye(d)], dim=1), torch.cat([-torch.eye(d), -torch.eye(d)], dim=1), torch.cat([-z_seq, torch.zeros(n, d)], dim=1), torch.cat([torch.zeros(1,d), torch.ones(1,d)], dim=1)], dim=0)
    #         b_ub = torch.cat([torch.zeros(2*d), torch.zeros(n)-1e-7, torch.ones(1)], dim=0)

    #         #b_ub[-1] = 1

    #         # res = linprog (-torch.cat([z_mean,torch.zeros_like(z_mean) ], dim=0), A_ub=A_ub, b_ub=b_ub, method='interior-point', bounds=(None,None))
    #         # #print(res)
    #         # print(res['message'])
    #         # w = res['x'][0:d]
    #         # w=torch.from_numpy(w).float()

    #         x = cp.Variable(d)
    #         objective = cp.Minimize(-z_mean @ x)
    #         constraints = [cp.norm(x,p=2) <= 1,-z_seq @ x <= torch.zeros(n)-1e-5]
    #         prob = cp.Problem(objective, constraints)
    #         result = prob.solve(solver="MOSEK")
    #         print(x)
    #         if d<15:
    #             print(x.value[0:d])
    #         else:
    #             print(x.value[0:15])
    #         #print(x.value[d:-1])
    #         #print(A_ub @ x.value )
    #         w=x.value
    #         w=torch.from_numpy(w).float()
            
    #         err_train = test_error(w, xs, ys)
    #         err = test_error(w, x_seq_test, y_seq_test)
    #         errs_train_avm_poly.append(err_train)
    #         errs_avm_poly.append(err)
    #         #print(w)
    #         print("w={}, train_err={}, err={}".format(b, err_train, err))

    #     wandb.log({"err_test_cavm": errs_avm_poly[0],
    #             "err_test_cmm": err_mm,
    #             "err_train_cavm": errs_train_avm_poly[0],
    #             "err_train_cmm": err_train_mm,
    #             })
        
    #     wmm_l2 =  wmm/torch.norm(wmm)
    #     w_l2= w/torch.norm(w)

    # if l1 is False:
    #     return err_mm, errs_avm_poly, err_train_mm, errs_train_avm_poly, None, None, None, None, wmm_l2, w_l2, None, None

    # else:
    #     # l1 margin
    #     print("====================l1=========================")
    #     # #clf = svm.LinearSVC(penalty='l1', loss='hinge', fit_intercept=False)
    #     # clf = svm.LinearSVC(penalty='l1', fit_intercept=False, dual=False, C=1e5, max_iter=int(1e8))
    #     # clf.fit(xs, ys)
    #     # wmm = -torch.Tensor(clf.coef_.flatten())
    #     # perf_train_mm_l1 = clf.score(xs, ys)
    #     # err_train_mm_l1 = 100*(1.-perf_train_mm_l1)
    #     # err_mm_l1 = test_error(wmm, x_seq_test, y_seq_test)

    #     _,_,wmm = solve_svc_problem(xs, ys, p=1)
    #     wmm = torch.Tensor(wmm.value)

    #     err_train_mm_l1 = test_error(wmm, xs, ys)
    #     err_mm_l1 = test_error(wmm, x_seq_test, y_seq_test)


    #     print("CMM train_err={}, err={}".format( err_train_mm_l1, err_mm_l1))

    #     errs_avm_poly_l1 = []
    #     errs_train_avm_poly_l1 = []

    #     for a in a_vals:

    #         b = tau**a

    #         # A_ub = torch.cat( [torch.cat([torch.eye(d), -torch.eye(d)], dim=1), torch.cat([-torch.eye(d), -torch.eye(d)], dim=1), torch.cat([-z_seq, torch.zeros(n, d)], dim=1), torch.cat([torch.zeros(1,d), torch.ones(1,d)], dim=1)], dim=0)
    #         # b_ub = torch.cat([torch.zeros(2*d), torch.zeros(n)-1e-7, torch.ones(1)], dim=0)
    #         # #b_ub[-1] = 1

    #         # # res = linprog (-torch.cat([z_mean,torch.zeros_like(z_mean) ], dim=0), A_ub=A_ub, b_ub=b_ub, method='interior-point', bounds=(None,None))
    #         # # #print(res)
    #         # # print(res['message'])
    #         # # w = res['x'][0:d]
    #         # # w=torch.from_numpy(w).float()

    #         # x = cp.Variable(2*d)
    #         # objective = cp.Minimize(-torch.cat([z_mean,torch.zeros_like(z_mean) ], dim=0) @ x)
    #         # constraints = [A_ub @ x <= b_ub]
    #         # prob = cp.Problem(objective, constraints)
    #         # result = prob.solve(verbose=True)
    #         # print(x)
    #         # if d<15:
    #         #     print(x.value[0:d])
    #         # else:
    #         #     print(x.value[0:15])
    #         # #print(x.value[d:-1])
    #         # #print(A_ub @ x.value )
    #         # w=x.value[0:d]

    #         x = cp.Variable(d)
    #         objective = cp.Minimize(-z_mean @ x)
    #         constraints = [cp.norm(x, p=1)<= 1,-z_seq @ x <= torch.zeros(n)-1e-5]
    #         prob = cp.Problem(objective, constraints)
    #         result = prob.solve(solver="MOSEK")

    #         w=x.value
    #         w=torch.from_numpy(w).float()

    #         #print(w)

    #         err_train_l1 = test_error(w, xs, ys)
    #         err_l1 = test_error(w, x_seq_test, y_seq_test)
    #         errs_train_avm_poly_l1.append(err_train_l1)
    #         errs_avm_poly_l1.append(err_l1)

    #         print("w={}, train_err={}, err={}".format(b, err_train_l1, err_l1))

    #     wandb.log({"err_test_cavm_l1": errs_avm_poly_l1[0],
    #             "err_test_cmm_l1": err_mm_l1,
    #             "err_train_cavm_l1": errs_train_avm_poly_l1[0],
    #             "err_train_cmm_l1": err_train_mm_l1,
    #             })
        
    #     wmm_l1 = wmm/torch.norm(wmm)
    #     w_l1 = w/torch.norm(w)

    # wandb.finish()

    # if l1 and l2:
    #     return err_mm, errs_avm_poly, err_train_mm, errs_train_avm_poly, err_mm_l1, errs_avm_poly_l1, err_train_mm_l1, errs_train_avm_poly_l1, wmm_l2, w_l2, wmm_l1, w_l1
    # else:
    #     return None, None, None, None, err_mm_l1, errs_avm_poly_l1, err_train_mm_l1, errs_train_avm_poly_l1, None, None, wmm_l1, w_l1



# def margin_classifiers_perf(d=1000,n=100,approx_tau=1, SNR=10, n_test=1e4, s=None, random_flip_prob=0, l1=True, l2=False, mixture=False, outlier_strength=100, seed=0):

#     n1 = min(int(np.round(approx_tau * n/(1.+approx_tau))), n-1)
#     n2 = n - n1
#     n1, n2 = max(n1, n2), min(n1, n2)

#     tau = n1/n2

#     config = dict(
#     n_train= n,
#     n_test= n_test,
#     n_features= d,
#     tau = tau,
#     SNR = SNR,
#     s=s,
#     label_noise_prob= random_flip_prob,
#     data_mixture = mixture,
#     seed = seed,
#     )

#     wandb.init(project="margin-class", entity="gizem", config=config)

#     if mixture:
#         xs, ys, x_seq_test, y_seq_test, z1s, z2s = create_data_mixture(d,n1,n2,n_test, s=s, random_flip_prob=random_flip_prob, SNR=SNR, outlier_strenght=outlier_strength, seed=seed)
#     else:
#         xs, ys, x_seq_test, y_seq_test, z1s, z2s = create_data_sparse(d,n1,n2,n_test, s=s, random_flip_prob=random_flip_prob, seed=seed)
        
#         w_gt = torch.zeros(d)
#         w_gt[0:s] = 1/(s ** 0.5)

#     z_seq = torch.cat((z1s, z2s), dim=0)
#     z_mean = torch.mean(z_seq, dim=0)
    
#     a_vals = [0.]

#     if l2:
#         # l2 margin
#         # clf = svm.LinearSVC(loss='hinge', fit_intercept=False, C=1e5, max_iter=int(1e8))
#         # clf.fit(xs, ys)
#         # wmm = -torch.Tensor(clf.coef_.flatten())
#         # #print(wmm)
#         # perf_train_mm = clf.score(xs, ys)
#         # err_train_mm = 100*(1.-perf_train_mm)
#         # err_mm = test_error(wmm, x_seq_test, y_seq_test)

#         _,_, wmm = solve_svc_problem(xs, ys, p=2) 
#         #print(wmm.value)
#         wmm = torch.Tensor(wmm.value)
#         err_train_mm = test_error(wmm, xs,ys)
#         err_mm = test_error(wmm, x_seq_test, y_seq_test)

#         print("CMM train_err={}, err={}".format( err_train_mm, err_mm))

#         errs_avm_poly = []
#         errs_train_avm_poly = []

#         for a in a_vals:

#             b = tau**a

#             A_ub = torch.cat( [torch.cat([torch.eye(d), -torch.eye(d)], dim=1), torch.cat([-torch.eye(d), -torch.eye(d)], dim=1), torch.cat([-z_seq, torch.zeros(n, d)], dim=1), torch.cat([torch.zeros(1,d), torch.ones(1,d)], dim=1)], dim=0)
#             b_ub = torch.cat([torch.zeros(2*d), torch.zeros(n)-1e-7, torch.ones(1)], dim=0)

#             #b_ub[-1] = 1

#             # res = linprog (-torch.cat([z_mean,torch.zeros_like(z_mean) ], dim=0), A_ub=A_ub, b_ub=b_ub, method='interior-point', bounds=(None,None))
#             # #print(res)
#             # print(res['message'])
#             # w = res['x'][0:d]
#             # w=torch.from_numpy(w).float()

#             x = cp.Variable(d)
#             objective = cp.Minimize(-z_mean @ x)
#             constraints = [cp.norm(x,p=2) <= 1,-z_seq @ x <= torch.zeros(n)-1e-5]
#             prob = cp.Problem(objective, constraints)
#             result = prob.solve(solver="MOSEK")
#             print(x)
#             if d<15:
#                 print(x.value[0:d])
#             else:
#                 print(x.value[0:15])
#             #print(x.value[d:-1])
#             #print(A_ub @ x.value )
#             w=x.value
#             w=torch.from_numpy(w).float()
            
#             err_train = test_error(w, xs, ys)
#             err = test_error(w, x_seq_test, y_seq_test)
#             errs_train_avm_poly.append(err_train)
#             errs_avm_poly.append(err)
#             #print(w)
#             print("w={}, train_err={}, err={}".format(b, err_train, err))

#         wandb.log({"err_test_cavm": errs_avm_poly[0],
#                 "err_test_cmm": err_mm,
#                 "err_train_cavm": errs_train_avm_poly[0],
#                 "err_train_cmm": err_train_mm,
#                 })
        
#         wmm_l2 =  wmm/torch.norm(wmm)
#         w_l2= w/torch.norm(w)

#     if l1 is False:
#         return err_mm, errs_avm_poly, err_train_mm, errs_train_avm_poly, None, None, None, None, wmm_l2, w_l2, None, None

#     else:
#         # l1 margin
#         print("====================l1=========================")
#         # #clf = svm.LinearSVC(penalty='l1', loss='hinge', fit_intercept=False)
#         # clf = svm.LinearSVC(penalty='l1', fit_intercept=False, dual=False, C=1e5, max_iter=int(1e8))
#         # clf.fit(xs, ys)
#         # wmm = -torch.Tensor(clf.coef_.flatten())
#         # perf_train_mm_l1 = clf.score(xs, ys)
#         # err_train_mm_l1 = 100*(1.-perf_train_mm_l1)
#         # err_mm_l1 = test_error(wmm, x_seq_test, y_seq_test)

#         _,_,wmm = solve_svc_problem(xs, ys, p=1)
#         wmm = torch.Tensor(wmm.value)

#         err_train_mm_l1 = test_error(wmm, xs, ys)
#         err_mm_l1 = test_error(wmm, x_seq_test, y_seq_test)


#         print("CMM train_err={}, err={}".format( err_train_mm_l1, err_mm_l1))

#         errs_avm_poly_l1 = []
#         errs_train_avm_poly_l1 = []

#         for a in a_vals:

#             b = tau**a

#             # A_ub = torch.cat( [torch.cat([torch.eye(d), -torch.eye(d)], dim=1), torch.cat([-torch.eye(d), -torch.eye(d)], dim=1), torch.cat([-z_seq, torch.zeros(n, d)], dim=1), torch.cat([torch.zeros(1,d), torch.ones(1,d)], dim=1)], dim=0)
#             # b_ub = torch.cat([torch.zeros(2*d), torch.zeros(n)-1e-7, torch.ones(1)], dim=0)
#             # #b_ub[-1] = 1

#             # # res = linprog (-torch.cat([z_mean,torch.zeros_like(z_mean) ], dim=0), A_ub=A_ub, b_ub=b_ub, method='interior-point', bounds=(None,None))
#             # # #print(res)
#             # # print(res['message'])
#             # # w = res['x'][0:d]
#             # # w=torch.from_numpy(w).float()

#             # x = cp.Variable(2*d)
#             # objective = cp.Minimize(-torch.cat([z_mean,torch.zeros_like(z_mean) ], dim=0) @ x)
#             # constraints = [A_ub @ x <= b_ub]
#             # prob = cp.Problem(objective, constraints)
#             # result = prob.solve(verbose=True)
#             # print(x)
#             # if d<15:
#             #     print(x.value[0:d])
#             # else:
#             #     print(x.value[0:15])
#             # #print(x.value[d:-1])
#             # #print(A_ub @ x.value )
#             # w=x.value[0:d]

#             x = cp.Variable(d)
#             objective = cp.Minimize(-z_mean @ x)
#             constraints = [cp.norm(x, p=1)<= 1,-z_seq @ x <= torch.zeros(n)-1e-5]
#             prob = cp.Problem(objective, constraints)
#             result = prob.solve(solver="MOSEK")

#             w=x.value
#             w=torch.from_numpy(w).float()

#             #print(w)

#             err_train_l1 = test_error(w, xs, ys)
#             err_l1 = test_error(w, x_seq_test, y_seq_test)
#             errs_train_avm_poly_l1.append(err_train_l1)
#             errs_avm_poly_l1.append(err_l1)

#             print("w={}, train_err={}, err={}".format(b, err_train_l1, err_l1))

#         wandb.log({"err_test_cavm_l1": errs_avm_poly_l1[0],
#                 "err_test_cmm_l1": err_mm_l1,
#                 "err_train_cavm_l1": errs_train_avm_poly_l1[0],
#                 "err_train_cmm_l1": err_train_mm_l1,
#                 })
        
#         wmm_l1 = wmm/torch.norm(wmm)
#         w_l1 = w/torch.norm(w)

#     wandb.finish()

#     if l1 and l2:
#         return err_mm, errs_avm_poly, err_train_mm, errs_train_avm_poly, err_mm_l1, errs_avm_poly_l1, err_train_mm_l1, errs_train_avm_poly_l1, wmm_l2, w_l2, wmm_l1, w_l1
#     else:
#         return None, None, None, None, err_mm_l1, errs_avm_poly_l1, err_train_mm_l1, errs_train_avm_poly_l1, None, None, wmm_l1, w_l1

if __name__ == '__main__':
   main()
    