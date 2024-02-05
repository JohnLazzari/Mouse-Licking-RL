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

                grad = p.grad.data
                state = self.state[p]

                # Initialize state parameters if not present
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'] += 1
                beta1, beta2 = group['betas']

                state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * grad
                state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * (grad ** 2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                lr = group['lr'] * (np.sqrt(bias_correction2) / bias_correction1)
                p.data.addcdiv_(-lr, state['exp_avg'], torch.sqrt(state['exp_avg_sq']) + group['eps'])

                # Weight clipping
                if group['names'][i] == "weight_hh_l0":
                    p.data = torch.clamp(p.data, -10, group['inhib_clip_value'])
                    assert (p.data < 0).all()
                if group['names'][i] == "weight_ih_l0":
                    p.data = torch.clamp(p.data, group['excite_clip_value'], 10)
                    assert (p.data > 0).all()
                if group['names'][i] == "mean_linear.weight":
                    p.data = torch.clamp(p.data, -10, group['inhib_clip_value'])
                    assert (p.data < 0).all()

        return loss