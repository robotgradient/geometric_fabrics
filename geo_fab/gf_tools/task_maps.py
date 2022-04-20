import torch
from geo_fab.gf_tools import GeometricFabrics


class BaseMapFabric(GeometricFabrics):
    '''
    Relative Mapping in Fabrics. We apply a Mapping to change the center
    '''
    def __init__(self, F):
        super(RelMapFabric, self).__init__(F.finsler_energy.M, F.finsler_energy.epsilon,
                                               F.f, F.phi, F.B)
        self.F = F

    def map(self, q, dq):
        return q, dq

    def pullback(self, q, dq, ddq):
        return ddq

    def compute_values(self, q, dq):
        q, dq = self.map(q, dq)
        return self.F.compute_values(q, dq)

    def get_ddq(self, q, dq):
        q, dq = self.map(q, dq)
        ddq = self.F.get_ddq(q, dq)
        return self.pullback(q, dq, ddq)

    def get_energy(self, q, dq):
        q, dq = self.map(q, dq)
        return self.F.get_energy(q, dq)


class RelMapFabric(GeometricFabrics):
    '''
    Relative Mapping in Fabrics. We apply a Mapping to change the center
    '''
    def __init__(self, F, qd = torch.zeros(2)):
        super(RelMapFabric, self).__init__(F.finsler_energy.M, F.finsler_energy.epsilon,
                                               F.f, F.phi, F.B)
        self.qd = qd
        self.F = F

    def map(self, q, dq):
        phi = q - self.qd
        return phi, dq


class RepulsionMapFabric(GeometricFabrics):
    '''
    Relative Mapping in Fabrics. We apply a Mapping to change the center
    '''
    def __init__(self, F, q0 = torch.zeros(2), r=0.5):
        super(RelMapFabric, self).__init__(F.finsler_energy.M, F.finsler_energy.epsilon,
                                               F.f, F.phi, F.B)
        self.F = F

        self.q0 = q0
        self.r = r

    def map(self, q, dq):
        x_norm = torch.linalg.norm(q - self.q0)
        x = x_norm/self.r - 1.

        return x, dq

    def pullback(self, q, dq, ddq):
        return ddq


