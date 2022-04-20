import numpy as np
import torch

from geo_fab.envs import PointEnv

from geo_fab.gf_tools import metrics, potentials, geometric_fabrics, policies, task_maps

def pi(q,dq):
    out = torch.zeros_like(q)

    out[:, 1] = torch.ones_like(q[:,0])*100
    #out[:, 1] = torch.sin(2*q[:,0])*200.
    return out


##################
## Build Fabric ##
##################
M = metrics.NaiveMetric().M
eps = metrics.epsilon
f   = policies.NoPolicy().f
B   = potentials.NaiveDamping(torch.ones(2)).B
phi = potentials.AttractorPotential(k=100)

F = geometric_fabrics.GeometricFabrics(M=M, epsilon=eps, f=f,
                                       phi=phi, B=B)

F_energized = geometric_fabrics.EnergizingFabrics(F=F, pi=pi)
policy = task_maps.RelMapFabric(F=F_energized, qd=torch.ones(2)*5.)
##################

env = PointEnv()

T=10000
q0 = np.array([3., 3.])
dq0 = np.array([10., 0.])

q, dq =env.reset(q0= q0, dq0=dq0)
for t in range(T):

    q = torch.Tensor(q[None,:])
    dq = torch.Tensor(dq[None,:])
    ## Get action
    ddq = policy.get_ddq(q, dq)
    e = policy.get_energy(q,dq)
    print('Energy:', e)

    ddq = ddq.detach().cpu().numpy()

    ## Env dynamics step
    q, dq =env.step(ddq[0,:])