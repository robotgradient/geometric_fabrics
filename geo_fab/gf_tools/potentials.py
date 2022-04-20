import torch


class NoPotential():
    def __init__(self):
        pass

    def get_energy(self, q, dq):
        return torch.zeros_like(q[:,0])

    def get_grad(self, q, dq):
        return torch.zeros_like(dq)


class NaiveDamping():
    def __init__(self, B_diag=torch.zeros(2)):
        self.B_diag = B_diag

    def B(self, q, dq):
        return torch.diag(self.B_diag)


class AttractorPotential():

    def __init__(self, k=1., alpha  = 1.,):
        self.k = k
        self.alpha = alpha

    def get_energy(self, q, dq):
        x_norm = torch.linalg.norm(q)
        exp_norm_x = torch.exp(-2*self.alpha*x_norm)
        log_1_exp = torch.log(1 + exp_norm_x)
        phi = self.k * (x_norm + (1/self.alpha)*log_1_exp)
        return phi

    def get_grad(self, q, dq):
        q.requires_grad_(True)
        phi = self.get_energy(q, dq)

        grad_phi = torch.autograd.grad(phi, q, only_inputs=True)
        return grad_phi[0]


