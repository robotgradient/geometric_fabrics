import numpy as np
import torch


def M(q, dq):
    return torch.eye(q.shape[-1])

def epsilon(q, dq):
    return torch.zeros_like(q)


class EnergizingFabrics():
    '''
    Energizing Fabrics: Apply an acceleration in the perpendicular direction of the motion dq.T \cdot f = 0
    This acceleration bends the direction of the acceleration without changing the total energy.
    '''

    def __init__(self, pi, M=M, epsilon=epsilon, dim=2):
        self.dim = 2
        self.M = M
        self.epsilon = epsilon
        self.pi = pi

    def compute_values(self, q, dq):
        ## Compute Metric and fictitious forces ##
        M = self.M(q,dq)
        epsilon = self.epsilon(q,dq)
        pi = self.pi(q,dq)
        ## Get metric-weighted projection
        _M12 = M**0.5
        _M12_inv = torch.inverse(_M12)
        v = torch.einsum('dm,bm->bd', _M12, dq)
        ## Deal with v_norm = 0 ##
        v_norm = torch.linalg.norm(v, dim=1) + 1e-12
        v_hat = v/v_norm[:,None]
        I_vv = torch.eye(self.dim) - v_hat.T@v_hat
        P_e = _M12@I_vv@_M12_inv

        f = torch.einsum('dm,bm->bd',P_e,
                         (torch.einsum('dm,bm->bd', M, pi) + epsilon))
        return M, epsilon, f

    def get_ddq(self, q,dq):
        M, epsilon, f = self.compute_values(q, dq)

        const_dyn = epsilon + f
        ddq = torch.einsum('dm,bm->bd', torch.inverse(M), const_dyn)
        return ddq

    def get_energy(self, q, dq):
        M, epsilon, f = self.compute_values(q, dq)

        K = 0.5* dq@M@dq.T
        return K







