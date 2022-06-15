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

    allowed_types = {"exp", "logit", "linear"}

    def __init__(self, type: str, alpha: float, reduction: str):
        super().__init__()
        self.type = type
        assert type in self.allowed_types
        self.alpha = float(alpha)
        assert reduction == "none"

    def margin_fn(self, margin_vals: torch.Tensor):
        indicator = margin_vals <= 1
        inv_part = torch.pow(
            margin_vals.abs(), -1 * self.alpha
        )  # prevent exponentiating negative numbers by fractional powers
        if self.type == "exp":
            exp_part = torch.exp(-1 * margin_vals)
            scores = exp_part * indicator + inv_part * (~indicator)
            return scores
        if self.type == "logit":
            indicator = margin_vals <= 1
            inv_part = torch.pow(margin_vals.abs(), -1 * self.alpha)
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
