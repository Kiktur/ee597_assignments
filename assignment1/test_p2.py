import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.integrate import quad

def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))

# Fading standard deviations in dB
sigma_vals_db = [5, 10, 20]
eta = 2

pt_db = 0      # Transmit power in dBm
kref_db = -20  # Path loss at reference distance in dB
d0 = 1         # Reference distance in m
N0 = -80       # Noise in dBm

# Distance vector
d = np.arange(1, 7001)

# Received power and SNR in dB (no fading)
pr_db = pt_db + kref_db - 10 * eta * np.log10(d / d0)
ebno_db = pr_db - N0

# Convert SNR to linear scale
ebno_lin = 10 ** (ebno_db / 10)

# BER without fading
BER_no_fading = qfunc(np.sqrt(2 * ebno_lin))

plt.figure()
plt.semilogy(d, BER_no_fading, 'k', linewidth=2, label='No fading')

BER_avg = np.zeros((len(sigma_vals_db), len(d)))

for k, sigma_db in enumerate(sigma_vals_db):
    sigma_lin = (10 ** (sigma_db / 10) - 1)  # Approx linear scale variance (more accurate)
    
    for i in range(len(d)):
        gamma_bar = ebno_lin[i]

        # Integrand in linear SNR space
        def integrand(x):
            # x is linear SNR deviation due to fading
            # Log-normal PDF in linear scale:
            mu = np.log(gamma_bar) - 0.5 * np.log(1 + sigma_lin**2)
            s = np.sqrt(np.log(1 + sigma_lin**2))
            pdf = (1 / (x * s * np.sqrt(2 * np.pi))) * np.exp(- (np.log(x) - mu)**2 / (2 * s**2))
            return qfunc(np.sqrt(2 * x)) * pdf

        # Integrate over a reasonable linear SNR range
        lower = gamma_bar * 1e-3
        upper = gamma_bar * 1e3
        BER_avg[k, i], _ = quad(integrand, lower, upper)

    plt.semilogy(d, BER_avg[k, :], linewidth=2, label=rf'$\sigma = {sigma_db}$ dB')

plt.xlabel('Distance (m)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('Expected BER vs Distance with Log-Normal Fading (Linear Integration)')
plt.ylim([1e-5, 1])
plt.grid(True, which='both')
plt.legend(loc='upper right')
plt.show()
