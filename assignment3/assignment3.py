import numpy as np

mu = 0 # zero mean


# Kernel is the same as covariance

sigma = 2

# Iterate over values of Tc to demonstrate fading
Tc = [0,1,2]

t = 0
t_prime = 0


K = (sigma**2) * np.exp((-1 * (t - t_prime)**2) / (Tc**2))


# Need to generate a covariance matrix and perform SVD to get X ~ N(0,K)