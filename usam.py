'''
to ease the usage in conjunction with various base optimizers, 
we slightly modified the original version of SAM optimizer in Pytorch (https://github.com/davda54/sam), 
and used the style of Lookahead optimizer implemented by Timm library 
(https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/lookahead.py).
'''

import torch
import torch.nn as nn
from collections import defaultdict

class USAM(torch.optim.Optimizer):
    def __init__(self, base_optimizer, rho=0.05, adaptive=False, stable=False):
        # NOTE super().__init__() not called on purpose
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, stable=stable)
        if stable:
            defaults.update({'exact_gradient_norm': 1.0, 'surrogate_gradient_norm': 1.0})
            
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.base_optimizer.param_groups:
                group.setdefault(name, default)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            if group['stable']: group['exact_gradient_norm'] = grad_norm.item()

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * group["rho"]
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        if self.param_groups[0]['stable']: grad_norm = self._grad_norm()
        for group in self.param_groups:
            if group['stable']:
                group['surrogate_gradient_norm'] = grad_norm.item()
                scale = group['exact_gradient_norm'] / (group['surrogate_gradient_norm'] + 1.0e-12)
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                if group['stable']: p.grad.mul_(scale) # downscale the gradient magnitude to the same as the exact gradient
                # if group['stable']: p.grad.mul_(min(1, scale)) # e.g. explicitly truncate the upper bound to further enhance stability

        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        
        self.first_step(zero_grad=True)     
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
