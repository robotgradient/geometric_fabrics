import torch


class NoPolicy():
    def __init__(self):
        pass

    def f(self, q, dq):
        return torch.zeros_like(dq)