import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc # for Q function


# Again consider log-normal fading with 𝜎dB. Assuming that an SNR below 10dB results in an
# outage event (i.e. is unacceptable), plot the outage probability (not in log-scale) as a function of
# distance as three separate curves for 𝜎dB = 5dB, 10dB and 20dB (again, assuming η = 2 and
# using all the other parameters the same as #2 above).



def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))


eta = 2
sigma_vals = [5, 10, 20]

pt_db = 0
kref_db = -20
d0 = 1
N0 = -80

d = np.arange(1, 7001)

pr_db = pt_db + kref_db - 10 * eta * np.log10(d / d0)
ebno_db = pr_db - N0

threshold_db = 10  # outage threshold

plt.figure()

for sigma in sigma_vals:
    qfunc_exp = (threshold_db - ebno_db) / sigma
    pout = 1 - qfunc(qfunc_exp)
    plt.plot(d, pout, linewidth=2, label=rf'$\sigma = {sigma}$ dB')


# TODO: Could make plot look nicer??
plt.xlabel('Distance (m)')
plt.ylabel('Probability of Outage')
plt.title('Probability of Outage vs Distance (Threshold = 10 dB)')
plt.grid(True)
plt.legend(loc='lower left')
plt.show()
