import torch
import torch.optim as optim
import numpy as np

class CustomAdamOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, excite_clip_value=1e-3, names=None):
        defaults = dict(lr=lr, betas=betas, eps=eps, excite_clip_value=excite_clip_value, names=names)
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
                
                # Initialize state parameters
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                eps = group['eps']
                lr = group['lr']
                state['step'] += 1

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(1 - beta1, grad)
                v.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Bias correction
                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])

                # Update parameters
                p.data.add_(-lr, m_hat / (torch.sqrt(v_hat) + eps))

                # Weight clipping
                if group['names'][i] == "weight_ih_l0":
                    p.data.clamp_(group['excite_clip_value'], 10)
                    assert (p.data >= 0).all()

        return loss