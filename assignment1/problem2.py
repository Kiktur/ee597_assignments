import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc # for Q function
from scipy.integrate import quad # for integration


# Now consider a wireless channel with log-normal fading with a given standard deviation 𝜎dB .
# Numerically calculate and plot the expected BER as a function of distance (set η = 2, using all
# the other parameters the same as #2 above) as 3 separate curves for 𝜎dB = 5dB, 10dB and 20dB,
# and on the same plot include your plot from #2 above. How do the curves compare? Comment on
# the plot.
# Hint: you can do this plot without doing simulations. Use the integral definition of E[BER]
# (which is the integral of the product of the BER curve and the fading distribution), and use
# a numerical method to calculate the integral.


def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))

sigma_vals = [5, 10, 20]
eta = 2

pt_db = 0
kref_db = -20
d0 = 1
N0 = -80

# 1 to 7000 meters
d = np.arange(1, 7001)

pr_db = pt_db + kref_db - 10 * eta * np.log10(d / d0)
ebno_db = pr_db - N0

# Convert to linear
ebno_lin = 10 ** (ebno_db / 10)

BER_no_fading = qfunc(np.sqrt(2 * ebno_lin))

plt.figure()
plt.semilogy(d, BER_no_fading, 'k', linewidth=2, label='No fading')

BER_avg = np.zeros((len(sigma_vals), len(d)))

for k, sigma in enumerate(sigma_vals):
    for i in range(len(d)):
        gamma_bar_dB = ebno_db[i]

        # ebno (or the SNR) is different at every point due to received power being different, and the BER calculation depends on each SNR
        # Therefore, integral must be calculated at each point
        def integrand(x):
            return ( # f(gamma) is the qfunc portion, and g(gamma) (i.e. log fading portion) is everything else
                qfunc(np.sqrt(2 * 10 ** ((gamma_bar_dB + x) / 10))) * (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-x ** 2 / (2 * sigma ** 2))
            )

        # Underscore ignores the error result that comes from quad()
        BER_avg[k, i], _ = quad(integrand, -5 * sigma, 5 * sigma)

    plt.plot(d, BER_avg[k, :], linewidth=2, label=rf'$\sigma = {sigma}$ dB')


# TODO: Maybe make the plot look nicer and increase distance range
plt.xlabel('Distance (m)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('Average BER vs Distance with Log-Normal Fading')
plt.ylim([1e-5, 1])
plt.yscale('linear')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()
