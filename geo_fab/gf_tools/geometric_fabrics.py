import numpy as np
import torch

class BaseGeoFabric():
    '''
    Base Geometric Fabric: A base class to represent reactive motion generators based on geometric fabrics
    '''

    def __init__(self):
        pass

    def get_ddq(self, q,dq):
        print('To be define')

    def get_energy(self, q, dq):
        print('To be define')


class FinslerEnergy():
    '''
    Finsler Energy: Apply an acceleration in the perpendicular direction of the motion dq.T \cdot f = 0
    This acceleration bends the direction of the acceleration without changing the total energy.
    '''

    def __init__(self, M, epsilon):
        self.M = M
        self.epsilon = epsilon

    def compute_values(self, q, dq):
        metric = self.M(q, dq)
        eps = self.epsilon(q,dq)
        return metric, eps

    def get_energy(self, q, dq):
        metric, eps = self.compute_values(q, dq)
        K = 0.5 * dq @ metric @ dq.T
        return K


class GeometricFabrics(BaseGeoFabric):
    '''
    Geometric Fabrics: Apply an acceleration in the perpendicular direction of the motion dq.T \cdot f = 0
    This acceleration bends the direction of the acceleration without changing the total energy.
    '''

    def __init__(self, M, epsilon, f, phi, B, dim=2):
        super(GeometricFabrics, self).__init__()
        self.dim = dim
        self.finsler_energy = FinslerEnergy(M, epsilon)
        self.f = f
        self.phi = phi
        self.B = B

    def compute_values(self, q, dq):
        ## Compute Metric and fictitious forces ##
        M, epsilon = self.finsler_energy.compute_values(q, dq)
        ## Compute bending force, grad_phi and B
        f_b = self.f(q, dq)
        grad_phi = self.phi.get_grad(q, dq)
        B = self.B(q, dq)
        return M, epsilon, f_b, grad_phi, B

    def get_ddq(self, q,dq):
        M, epsilon, f_b, grad_phi, B = self.compute_values(q, dq)

        const_dyn = epsilon + f_b
        energy_pump = -grad_phi
        energy_release = -torch.einsum('mn,bn->bm', B, dq)
        force = energy_pump + energy_release - const_dyn

        ddq = torch.einsum('dm,bm->bd', torch.inverse(M), force)
        return ddq

    def get_energy(self, q, dq):
        L_e = self.finsler_energy.get_energy(q, dq)
        phi = self.phi.get_energy(q, dq)
        return L_e + phi


class EnergizingFabrics(GeometricFabrics):
    '''
    Energizing Fabrics: Apply an acceleration in the perpendicular direction of the motion dq.T \cdot f = 0
    This acceleration bends the direction of the acceleration without changing the total energy.
    '''

    def __init__(self, pi, F, dim=2):
        super(EnergizingFabrics, self).__init__(F.finsler_energy.M, F.finsler_energy.epsilon,
                                               F.f, F.phi, F.B)
        self.dim = dim
        self.F = F
        self.pi = pi

    def compute_values(self, q, dq):
        ## Compute Metric and fictitious forces ##
        M, epsilon, f_b, grad_phi, B = self.F.compute_values(q, dq)
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

        f_b = torch.einsum('dm,bm->bd',P_e,
                         (torch.einsum('dm,bm->bd', M, pi) + epsilon + f_b))
        return M, epsilon, f_b, grad_phi, B


class ComposedGeometricFabrics(BaseGeoFabric):
    '''
    Composed Geometric Fabric: A weighted sum of the Geometric Fabrics
    '''
    def __init__(self, Fs, ws):
        self.Fs = Fs
        self.ws = ws

    def get_ddq(self, q,dq):
        print('To be define')

    def get_energy(self, q, dq):
        E = 0
        for Fi, wi in zip(self.Fs, self.ws):
            E += wi*Fi.get_energy(q, dq)
        return E


