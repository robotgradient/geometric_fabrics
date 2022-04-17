import numpy as np
from geo_fab.envs import PointEnv



env = PointEnv()

T=10000
s=env.reset()
for t in range(T):
    a=np.random.randn(2)
    s=env.step(a)

