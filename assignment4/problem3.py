import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Simplified optimality condition (eq. 1 with pi=p for all i, sigma=1):
#   T*(n*p - 1) + (T-1)*(1-p)^n = 0
# For T=1 this gives p* = 1/n directly.

def optimal_p(n, T):
    if T == 1:
        return 1.0 / n
    f = lambda p: T * (n * p - 1) + (T - 1) * (1 - p) ** n
    # p* < 1/n for T > 1; search in (0, 1/n)
    return brentq(f, 1e-9, 1.0 / n - 1e-12)

def throughput(n, p, T):
    return n * p * (1 - p) ** (n - 1) * T / ((1 - p) ** n + (1 - (1 - p) ** n) * T)

ns = np.arange(2, 21)
T_values = [1, 10, 100]
colors = ['tab:blue', 'tab:orange', 'tab:green']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

for T, c in zip(T_values, colors):
    p_star = np.array([optimal_p(n, T) for n in ns])
    S_total = np.array([throughput(n, p_star[i], T) for i, n in enumerate(ns)])
    ax1.plot(ns, p_star, '-o', color=c, markersize=5, label=f'T = {T}')
    ax2.plot(ns, S_total, '-o', color=c, markersize=5, label=f'T = {T}')

ax1.set_ylabel('Optimal $p^*$', fontsize=12)
ax1.set_title('Optimal Contention Probability and Throughput vs. Number of Nodes\n(p-persistent CSMA, $\\sigma=1$)', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Number of nodes $n$', fontsize=12)
ax2.set_ylabel('Total throughput $S$', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assignment4/problem3.png', dpi=150, bbox_inches='tight')
print("Saved assignment4/problem3.png")
plt.show()
