import torch
import math
import torch.nn as nn


class PolynomialLoss_pure(nn.Module):
    """
    Poly-tailed margin based losses that decay as v^{-\alpha} for \alpha > 0.
    The theory here is that poly-tailed losses do not have max-margin behavior
    and thus can work with importance weighting.

    Poly-tailed losses are not defined at v=0 for v^{-\alpha}, and so there are
    several variants that are supported via the [[type]] option
    exp : f(v):= exp(-v+1) for v < 1, 1/v^\alpha otherwise
    logit: f(v):= 1/log(2)log(1+exp(-v+1)) for v < 1, 1/v^\alpha else.
    """

    allowed_types = {"exp", "logit", "linear"}

    def __init__(self, type: str, alpha: float =1 , beta: float = 1,  reduction: str = 'none'):
        super().__init__()
        self.type = type
        assert type in self.allowed_types
        self.alpha = float(alpha)
        self.beta = float(beta)
        assert reduction == "none"


    def forward(self, logits, target):
        target_sign = 2 * target - 1
        margin_scores = (logits[:, 1] - logits[:, 0]) * target_sign
        loss_values = torch.pow((margin_scores-(self.beta-1)).abs(), -1 * self.alpha)
        return loss_values


class PolynomialLoss(nn.Module):
    """
    Poly-tailed margin based losses that decay as v^{-\alpha} for \alpha > 0.
    The theory here is that poly-tailed losses do not have max-margin behavior
    and thus can work with importance weighting.

    Poly-tailed losses are not defined at v=0 for v^{-\alpha}, and so there are
    several variants that are supported via the [[type]] option
    exp : f(v):= exp(-v+1) for v < 1, 1/v^\alpha otherwise
    logit: f(v):= 1/log(2)log(1+exp(-v+1)) for v < 1, 1/v^\alpha else.
    """

    allowed_types = {"exp", "logit", "linear"}

    def __init__(self, type: str, alpha: float =1 , beta: float = 1,  reduction: str = 'none'):
        super().__init__()
        self.type = type
        assert type in self.allowed_types
        self.alpha = float(alpha)
        self.beta = float(beta)
        assert reduction == "none"

    def margin_fn(self, margin_vals: torch.Tensor):
        indicator = margin_vals <= self.beta
        inv_part = torch.pow(
            (margin_vals-(self.beta-1)).abs(), -1 * self.alpha
        )  # prevent exponentiating negative numbers by fractional powers
        if self.type == "exp":
            exp_part = torch.exp(-1 * margin_vals)
            scores = exp_part * indicator + inv_part * (~indicator)
            return scores

        if self.type == "logit":
            indicator = margin_vals <= self.beta
            inv_part = torch.pow((margin_vals-(self.beta-1)).abs(), -1 * self.alpha)
            logit_inner = -1 * margin_vals
            logit_part = torch.nn.functional.softplus(logit_inner) / (
                math.log(1 + math.exp(-1))
            )
            scores = logit_part * indicator + inv_part * (~indicator)
            return scores
            
        if self.type == "linear":
            assert self.alpha > 1 
            linear_part = -1 * margin_vals + torch.ones_like(margin_vals) * (
                self.alpha / (self.alpha - 1)
            )
            scores = linear_part * indicator + inv_part * (~indicator) / (
                self.alpha - 1
            )
            return scores


    def forward(self, logits, target):
        target_sign = 2 * target - 1
        margin_scores = (logits[:, 1] - logits[:, 0]) * target_sign
        loss_values = self.margin_fn(margin_scores)
        return loss_values

class AverageMarginlLoss(nn.Module):
    """

    """

    def __init__(self, type: str):
        super().__init__()
        self.type = type

    def forward(self, logits, target):
        target_sign = 2 * target - 1
        margin_scores = (logits[:, 1] - logits[:, 0]) * target_sign

        return -margin_scores

class AverageMarginlLoss_sum(nn.Module):
    """

    """

    def __init__(self, type: str):
        super().__init__()
        self.type = type

    def forward(self, logits, target):
        margin_scores = logits.shape[1] * logits[range(target.shape[0]), target] - torch.sum(logits, dim=1)

        return -margin_scores

class AverageMarginlLoss_max(nn.Module):
    """

    """

    def __init__(self, type: str):
        super().__init__()
        self.type = type

    def forward(self, logits, target):
        tmp_logits = logits.clone()
        tmp_logits[range(target.shape[0]),target] = float("-Inf") 

        margin_scores = logits[range(target.shape[0]), target] - torch.max(tmp_logits, dim=1).values

        return -margin_scores

class AverageMarginlLoss_max_hinge(nn.Module):
    """

    """

    def __init__(self, type: str):
        super().__init__()
        self.type = type

    def forward(self, logits, target):
        tmp_logits = logits.clone()
        tmp_logits[range(target.shape[0]),target] = float("-Inf") 

        margin_scores = logits[range(target.shape[0]), target] - torch.max(tmp_logits, dim=1).values

        return torch.max(torch.zeros_like(margin_scores), 1-margin_scores)

class MCPolynomialLoss_max(nn.Module):
    """
    Poly-tailed margin based losses that decay as v^{-\alpha} for \alpha > 0.
    The theory here is that poly-tailed losses do not have max-margin behavior
    and thus can work with importance weighting.
    Poly-tailed losses are not defined at v=0 for v^{-\alpha}, and so there are
    several variants that are supported via the [[type]] option
    exp : f(v):= exp(-v+1) for v < 1, 1/v^\alpha otherwise
    logit: f(v):= 1/log(2)log(1+exp(-v+1)) for v < 1, 1/v^\alpha else.
    """

    allowed_types = {"logit"}

    def __init__(self, type: str, alpha: float, beta: float = 1 , reduction: str = "none"):
        super().__init__()
        self.type = type
        assert type in self.allowed_types
        self.alpha = float(alpha)
        self.beta = float(beta)
        assert reduction == "none"

    def margin_fn(self, margin_vals: torch.Tensor, logits:torch.Tensor, target:torch.Tensor):
        indicator = margin_vals <= self.beta
        inv_part = torch.pow((margin_vals-(self.beta-1)).abs(),-1*self.alpha)  # prevent exponentiating negative numbers by fractional powers
        if self.type == "logit":
            indicator = margin_vals <= self.beta
            #print(torch.mean(margin_vals), torch.min(margin_vals), torch.max(margin_vals))
            #print(indicator)
            #print(self.beta)
            inv_part = torch.pow((margin_vals-(self.beta-1)).abs(),-1*self.alpha)
            logit_inner = -1 * margin_vals
            #logit_part = torch.nn.functional.softplus(logit_inner)/(math.log(1+math.exp(-1)))
            logit_part = torch.nn.functional.cross_entropy(logits, target)
            scores = logit_part * indicator + inv_part * (~indicator)
            return scores

    def forward(self, logits, target):
        #target_sign = 2 * target - 1
        #margin_scores = (logits[:, 1] - logits[:, 0]) * target_sign
        #print(logits[:, target])
        #margin_scores = logits.shape[1]*logits[:, target] - torch.sum(logits, dim=1)
        
        tmp_logits = logits.clone()
        tmp_logits[range(target.shape[0]),target] = float("-Inf") 

        margin_scores = logits[range(target.shape[0]), target] - torch.max(tmp_logits, dim=1).values
        #margin_scores = logits[:, target]/torch.sum(logits, dim=1)
        loss_values = self.margin_fn(margin_scores, logits, target)
        return loss_values

class MCPolynomialLoss_sum(nn.Module):
    """
    Poly-tailed margin based losses that decay as v^{-\alpha} for \alpha > 0.
    The theory here is that poly-tailed losses do not have max-margin behavior
    and thus can work with importance weighting.
    Poly-tailed losses are not defined at v=0 for v^{-\alpha}, and so there are
    several variants that are supported via the [[type]] option
    exp : f(v):= exp(-v+1) for v < 1, 1/v^\alpha otherwise
    logit: f(v):= 1/log(2)log(1+exp(-v+1)) for v < 1, 1/v^\alpha else.
    """

    allowed_types = {"logit"}

    def __init__(self, type: str, alpha: float, beta: float = 1, reduction: str = "none"):
        super().__init__()
        self.type = type
        assert type in self.allowed_types
        self.alpha = float(alpha)
        self.beta = float(beta)
        assert reduction == "none"

    def margin_fn(self, margin_vals: torch.Tensor):
        indicator = margin_vals <= self.beta
        inv_part = torch.pow((margin_vals-(self.beta-1)).abs(),-1*self.alpha)  # prevent exponentiating negative numbers by fractional powers
        if self.type == "logit":
            indicator = margin_vals <= self.beta
            inv_part = torch.pow((margin_vals-(self.beta-1)).abs(),-1*self.alpha)
            logit_inner = -1 * margin_vals
            logit_part = torch.nn.functional.softplus(logit_inner)/(math.log(1+math.exp(-1)))
            scores = logit_part * indicator + inv_part * (~indicator)
            return scores

    def forward(self, logits, target):
        #target_sign = 2 * target - 1
        #margin_scores = (logits[:, 1] - logits[:, 0]) * target_sign
        #print(logits[:, target])
        #margin_scores = logits.shape[1]*logits[:, target] - torch.sum(logits, dim=1)
        
        margin_scores = 10 * logits[range(target.shape[0]), target] - torch.sum(logits, dim=1)
        #margin_scores = logits[:, target]/torch.sum(logits, dim=1)
        loss_values = self.margin_fn(margin_scores)
        return loss_values
