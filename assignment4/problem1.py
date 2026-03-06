import numpy as np
import matplotlib.pyplot as plt

# Parameters
g2_n2 = 1.0          # SNR gain per unit power for channel 2
g1_n1 = 2.0 * g2_n2  # g1/n1 = 2*(g2/n2), so channel 1 is better
Ptotal = 10.0         # total power budget

# Sweep P1 from 0 to Ptotal (100 discrete points for scatter plot)
N = 100
P1_vals = np.linspace(0, Ptotal, N)
P2_vals = Ptotal - P1_vals

# Shannon rates
R1 = np.log2(1 + P1_vals * g1_n1)
R2 = np.log2(1 + P2_vals * g2_n2)

sum_rate  = R1 + R2
rate_diff = R2 - R1

# Color points by P1 to show how power allocation drives the tradeoff
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(rate_diff, sum_rate, c=P1_vals, cmap='plasma', s=40, edgecolors='k', linewidths=0.3)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('P1 (power allocated to channel 1)')

# Mark the equal-power point (P1 = P2 = Ptotal/2)
idx_eq = N // 2
ax.scatter(rate_diff[idx_eq], sum_rate[idx_eq], marker='*', s=200, color='cyan',
           zorder=5, label=f'Equal power (P1=P2={Ptotal/2})')

# Mark the max sum-rate point
idx_max = np.argmax(sum_rate)
ax.scatter(rate_diff[idx_max], sum_rate[idx_max], marker='D', s=100, color='lime',
           zorder=5, label=f'Max sum-rate (P1={P1_vals[idx_max]:.1f})')

ax.set_xlabel('Rate Difference  $R_2 - R_1$  (bits/s/Hz)', fontsize=12)
ax.set_ylabel('Sum Rate  $R_1 + R_2$  (bits/s/Hz)', fontsize=12)
ax.set_title(
    f'Sum Rate vs Rate Difference\n'
    f'$g_1/n_1={g1_n1}$, $g_2/n_2={g2_n2}$, $P_{{total}}={Ptotal}$',
    fontsize=13
)
ax.legend()
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('assignment4/problem1.png', dpi=150, bbox_inches='tight')
print("Saved assignment4/problem1.png")
plt.show()

# Print a few key values for commentary
print(f"\nMax sum rate: {sum_rate.max():.4f} bpcu  at P1={P1_vals[idx_max]:.2f}, R2-R1={rate_diff[idx_max]:.4f}")
print(f"Equal power:  sum={sum_rate[idx_eq]:.4f} bpcu, R2-R1={rate_diff[idx_eq]:.4f}")
print(f"P1=0 (all to ch2): sum={sum_rate[0]:.4f}, R2-R1={rate_diff[0]:.4f}")
print(f"P2=0 (all to ch1): sum={sum_rate[-1]:.4f}, R2-R1={rate_diff[-1]:.4f}")
