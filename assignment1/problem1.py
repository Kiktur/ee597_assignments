import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc # for Q function

# Using the simple path loss model, and assuming SNR = Eb/No (i.e. the spectral efficiency R/W =
# 1), plot the Bit Error Rate of BPSK (in log-scale) as a function of distance if PT = 0 dBm, Kref, dB =
# -20dB, d0 = 1m, for η = 2 and η = 4; assume that noise power = -80dBm. Comment on the plot.
# Include distances sufficiently large to let the BER rise to close to 0.5.


def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))


pt_db = 0
kref_db = -20
d0 = 1

d = np.arange(1, 7001)  # 1 to 7000 meters

eta_vals = [2, 4]
N0 = -80

plt.figure()
for eta in eta_vals:
    # Simple path loss model
    pr_db = pt_db + kref_db - eta * 10 * np.log10(d / d0)

    # ebno in db
    ebno_db = pr_db - N0

    # Convert to linear
    ebno_lin = 10 ** (ebno_db / 10)

    # BER for BPSK
    BER = qfunc(np.sqrt(2 * ebno_lin))

    # Plot logarithmically
    plt.semilogy(d, BER, linewidth=2, label=rf'$\eta = {eta}$')


plt.xlabel('Distance (m)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs Distance')
# plt.legend([r'$\eta = 2$', r'$\eta = 4$'], loc='lower left')
plt.legend(loc='lower left')
plt.yscale('linear')
plt.ylim([1e-5, 1])
plt.grid(True)
plt.show()
