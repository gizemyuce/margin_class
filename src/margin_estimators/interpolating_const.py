import cvxpy as cp
import numpy as np
import torch

#from svc import solve_svc_problem
from src.margin_estimators.svc import *

def l2_average_interp(z1s, z2s):
    d = z1s.size(dim=1)

    z_seq = torch.cat((z1s, z2s), dim=0)
    z_mean = torch.mean(z_seq, dim=0)

    d = z_seq.size(dim=1)
    n = z_seq.size(dim=0)

    x = cp.Variable(d)
    objective = cp.Minimize(-z_mean @ x)
    constraints = [cp.norm(x,p=2) <= 1,-z_seq @ x <= torch.zeros(n)-1e-5]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver="MOSEK")

    print(x)
    if d<15:
        print(x.value[0:d])
    else:
        print(x.value[0:15])

    #print(x.value[d:-1])
    #print(A_ub @ x.value )
    w=x.value
    w=torch.from_numpy(w).float()

    return w/torch.norm(w)

def l1_average_interp(z1s, z2s, linear_program=True):
    

    z_seq = torch.cat((z1s, z2s), dim=0)
    z_mean = torch.mean(z_seq, dim=0)

    d = z_seq.size(dim=1)
    n = z_seq.size(dim=0)

    if linear_program:
        A_ub = torch.cat( [torch.cat([torch.eye(d), -torch.eye(d)], dim=1), torch.cat([-torch.eye(d), -torch.eye(d)], dim=1), torch.cat([-z_seq, torch.zeros(n, d)], dim=1), torch.cat([torch.zeros(1,d), torch.ones(1,d)], dim=1)], dim=0)
        b_ub = torch.cat([torch.zeros(2*d), torch.zeros(n)-1e-7, torch.ones(1)], dim=0)
        #b_ub[-1] = 1

        # res = linprog (-torch.cat([z_mean,torch.zeros_like(z_mean) ], dim=0), A_ub=A_ub, b_ub=b_ub, method='interior-point', bounds=(None,None))
        # #print(res)
        # print(res['message'])
        # w = res['x'][0:d]
        # w=torch.from_numpy(w).float()

        x = cp.Variable(2*d)
        objective = cp.Minimize(-torch.cat([z_mean,torch.zeros_like(z_mean) ], dim=0) @ x)
        constraints = [A_ub @ x <= b_ub]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver="MOSEK")

        print(x)
        if d<15:
            print(x.value[0:d])
        else:
            print(x.value[0:15])
        #print(x.value[d:-1])
        #print(A_ub @ x.value )
        w=x.value[0:d]

    else:
        x = cp.Variable(d)
        objective = cp.Minimize(-z_mean @ x)
        constraints = [cp.norm(x, p=1)<= 1,-z_seq @ x <= torch.zeros(n)-1e-5]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver="MOSEK")

        w=x.value
    
    w=torch.from_numpy(w).float()
    return w/torch.norm(w)

def l2_min_margin(xs, ys):
    _,_, wmm = solve_svc_problem(xs, ys, p=2) 
    wmm = torch.Tensor(wmm.value)
    return wmm/torch.norm(wmm)

def l1_min_margin(xs, ys):
    _,_,wmm = solve_svc_problem(xs, ys, p=1)
    wmm = torch.Tensor(wmm.value)
    return wmm/torch.norm(wmm)