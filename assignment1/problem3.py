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


pt_db = 0
kref_db = -20
d0 = 1
eta = 2
N0 = -80

d = np.arange(1, 7001)

pr_db = pt_db + kref_db - 10 * eta * np.log10(d / d0)
ebno_db = pr_db - N0
ebno_lin = 10 ** (ebno_db / 10)

W = 20 * (10 ** 6)

# Shannon capacity
C = W * np.log2(1 + ebno_lin)


plt.figure()
plt.plot(d, C, linewidth=2)
plt.grid(True)
plt.xlabel('Distance (m)')
plt.ylabel('Data Rate (bits/sec)')
plt.title('Maximum Data Rate vs Distance')
plt.show()
