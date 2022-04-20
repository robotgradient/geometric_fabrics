import torch



class AttractorMetric():
    '''
    Metric from Section VI. Particle Experiments in
    Geometric Fabrics: Generalizing Classical Mechanics to Capture the Physics of Behavior
    '''
    def __init__(self, lm=0., hm=1., alpha_m=0.1):
        self.lm = lm
        self.hm = hm
        self.alpha_m = alpha_m

    def M(self, q, dq):
        d_m = self.hm - self.lm
        x_norm = torch.linalg.norm(q)
        _K = d_m*torch.exp(-(self.alpha_m*x_norm)**2)
        G = _K*torch.eye(q.shape[-1]) + self.lm*torch.eye(q.shape[-1])
        return G

class NaiveMetric():
    '''
    Metric from Section VI. Particle Experiments in
    Geometric Fabrics: Generalizing Classical Mechanics to Capture the Physics of Behavior
    '''
    def __init__(self):
        pass

    def M(self, q, dq):
        return torch.eye(q.shape[-1])


def epsilon(q, dq):
    return torch.zeros_like(q)
