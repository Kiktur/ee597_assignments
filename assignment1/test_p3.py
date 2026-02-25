import numpy as np
import matplotlib.pyplot as plt

# Radio parameters
pt_db = 0        # Transmit power in dBm
N0_db = -80      # Noise power in dBm
eta = 2          # Path loss exponent
d0 = 1           # Reference distance in m
W = 20e6         # Channel bandwidth, 20 MHz
kref_db = -20

# Distance vector
d = np.arange(1, 7001)

# Convert dBm to linear mW
pt_lin = 10**(pt_db / 10)
N0_lin = 10**(N0_db / 10)

# Received power in mW (simple path loss)
# Pr_lin = pt_lin * (d0 / d)**eta

pr_db = pt_db + kref_db - 10 * eta * np.log10(d / d0)

Pr_lin = 10 ** (pr_db / 10)

# SNR in linear scale
snr_lin = Pr_lin / N0_lin

# Shannon capacity
C_shannon = W * np.log2(1 + snr_lin)

# Simplified approximate model: C ~ K / d^eta
K = W * pt_lin / (N0_lin * np.log(2)) * d0**eta
C_approx = K / d**eta

# Plot
plt.figure(figsize=(8,5))
plt.plot(d, C_shannon / 1e6, linewidth=2, label='Shannon Capacity')
plt.plot(d, C_approx / 1e6, '--', linewidth=2, label='Approx. $C \\propto d^{-\\eta}$')
plt.grid(True)
plt.xlabel('Distance (m)')
plt.ylabel('Data Rate (Mbps)')
plt.title('Maximum Data Rate vs Distance')
plt.legend()
plt.ylim([0, np.max(C_shannon)/1e6*1.1])
plt.show()
