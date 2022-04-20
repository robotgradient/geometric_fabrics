import torch
from geo_fab.gf_tools import GeometricFabrics



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

