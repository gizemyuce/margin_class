import torch
import math
import torch.nn as nn

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

    allowed_types = {"logit"}

    def __init__(self, type: str, alpha: float, beta: float = 1, reduction: str = "none"):
        super().__init__()
        self.type = type
        assert type in self.allowed_types
        self.alpha = float(alpha)
        assert reduction == "none"

    def margin_fn(self, margin_vals: torch.Tensor):
        indicator = margin_vals <= self.beta
        inv_part = torch.pow(
            (margin_vals-(self.beta-1)).abs(), -1 * self.alpha
        )  # prevent exponentiating negative numbers by fractional powers
        if self.type == "logit":
            indicator = margin_vals <= self.beta
            inv_part = torch.pow((margin_vals-(self.beta-1)).abs(),-1*self.alpha)
            logit_inner = -1 * margin_vals
            logit_part = torch.nn.functional.softplus(logit_inner)/(math.log(1+math.exp(-1)))
            scores = logit_part * indicator + inv_part * (~indicator)
            return scores

    def forward(self, logits, target):
        target_sign = 2 * target - 1
        margin_scores = (logits[:, 1] - logits[:, 0]) * target_sign
        loss_values = self.margin_fn(margin_scores)
        return loss_values

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
        assert reduction == "none"

    def margin_fn(self, margin_vals: torch.Tensor):
        indicator = margin_vals <= 1
        inv_part = torch.pow((margin_vals-(self.beta-1)).abs(),-1*self.alpha)  # prevent exponentiating negative numbers by fractional powers
        if self.type == "logit":
            indicator = margin_vals <= 1
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
        
        tmp_logits = logits.clone()
        tmp_logits[range(target.shape[0]),target] = float("-Inf") 

        margin_scores = logits[range(target.shape[0]), target] - torch.max(tmp_logits, dim=1).values
        #margin_scores = logits[:, target]/torch.sum(logits, dim=1)
        loss_values = self.margin_fn(margin_scores)
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
        assert reduction == "none"

    def margin_fn(self, margin_vals: torch.Tensor):
        indicator = margin_vals <= 1
        inv_part = torch.pow((margin_vals-(self.beta-1)).abs(),-1*self.alpha)  # prevent exponentiating negative numbers by fractional powers
        if self.type == "logit":
            indicator = margin_vals <= 1
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
