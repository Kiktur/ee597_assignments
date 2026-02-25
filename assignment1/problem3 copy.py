import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc # for Q function

# Using the Shannon capacity formula, plot the maximum data rate as a function of distance for η =
# 2 and all other radio parameters the same as question # 2 above (assuming the simple path loss
# model, no fading) if the channel bandwidth is 20 MHz. Comment on the plot. Additionally, try to
# come up with a simple but approximately correct equation to describe how the rate varies
# with distance and show your approximate curve also on the same plot. Note that this does
# not have one right answer, many approximate equations may exist, but try to make your
# equation as simple and general as possible (would be ideal if your equation can explicitly
# account for the parameters PT , η).


# TODO: Need to double check these results and/or make plot look nicer
# TODO: Make equation to describe rate variation with distance and approximate curve on same plot


def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))


pt_db = 0 # was 0
kref_db = -20
d0 = 1
eta = 2
N0 = -80

d = np.arange(1, 7001)

pr_db = pt_db + kref_db - 10 * eta * np.log10(d / d0)
ebno_db = pr_db - N0
ebno_lin = 10 ** (ebno_db / 10)

pt_lin = 10 ** (pt_db / 10)

N0_lin = 10 ** (N0/10)

W = 20e6


# Shannon capacity
C = W * np.log2(1 + ebno_lin)

# K = (W * pt_lin) / (d ** 0.004)
# K = (W * pt_lin) / (d ** 0.004)

# K = W * np.log2(1 + (pt_lin / (d**(1/eta)))) # solid
# K = W * np.log2(1 + ((pt_lin / N0_lin) / (d**(eta)))) # solid
# K = W * np.log2(1 + ((pt_lin) / (d**(eta)))) # solid
# K = W * np.log2(1 + (pt_lin * (1 / (d ** eta))))

# K = W * np.log2(1 + ((pt_lin) / ((d**(eta)) * N0_lin))) # solid x2
# K = W * np.log2(1 + ((pt_lin) / ((d**(eta)) * N0_lin))) # solid x2



# Approximate model: C ~ K / d^eta
# K = W * 10 ** ((pt_db + kref_db - N0) / 10) / np.log(2)
# C_approx = K / d**eta

# Simplified approximate model: C ~ K / d^eta
N0_lin = 10**(N0 / 10)
pt_lin = 10**(pt_db / 10)
K = W * pt_lin / (N0_lin * np.log(2)) * d0**eta
C_approx = K / d**eta



plt.figure(figsize=(8,5))
plt.plot(d, C / 1e6, linewidth=2)
# plt.plot(d, C_approx / 1e6, linewidth=2, label = f"Estimated model")
plt.plot(d, C_approx / 1e6, '--', linewidth=2, label='Approx. $C \\propto d^{-\\eta}$')
# plt.plot(d, y, linewidth=2, label=f"Estimated model")
# plt.plot(d, C_approx / 1e6, '--', linewidth=2, label='Approximate $C \\propto d^{-\\eta}$')
plt.grid(True)
plt.xlabel('Distance (m)')
plt.ylabel('Data Rate (Mbps)')
plt.title('Maximum Data Rate vs Distance')
plt.show()
