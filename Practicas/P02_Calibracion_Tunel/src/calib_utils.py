import numpy as np

def velocity_from_dp(dp, rho=1.225):
    dp = np.asarray(dp)
    return np.sqrt(2*dp/ rho)
