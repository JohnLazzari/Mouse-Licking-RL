import torch
import torch.optim as optim
import numpy as np

class CustomAdamOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, inhib_clip_value=-1e-3, excite_clip_value=1e-3, names=None):
        defaults = dict(lr=lr, betas=betas, eps=eps, inhib_clip_value=inhib_clip_value, excite_clip_value=excite_clip_value, names=names)
        super(CustomAdamOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)

                # Weight clipping
                if group['names'][i] == "weight_hh_l0":
                    p.data = torch.clamp(p.data, -10, group['inhib_clip_value'])
                    assert (p.data < 0).all()
                if group['names'][i] == "weight_ih_l0":
                    p.data = torch.clamp(p.data, group['excite_clip_value'], 10)
                    assert (p.data > 0).all()

        return loss