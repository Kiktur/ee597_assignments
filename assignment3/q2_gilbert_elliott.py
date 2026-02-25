import numpy as np
import matplotlib.pyplot as plt

def build_cov_matrix(t, sigma2, Tc):
    """Build covariance matrix: K(t,t') = sigma^2 * exp(-(t-t')^2 / Tc^2)"""
    diff = t[:, np.newaxis] - t[np.newaxis, :]
    return sigma2 * np.exp(-(diff ** 2) / (Tc ** 2))

def sample_gp(t, sigma2, Tc):
    """Sample one realization from a zero-mean GP."""
    K = build_cov_matrix(t, sigma2, Tc)
    return np.random.multivariate_normal(np.zeros(len(t)), K)

sigma2 = 1.0
S = 0.0
D = 1.0
num_slots = 500

# Time points at the beginning of each slot: t = n*D
t_slots = np.arange(num_slots) * D

# Sweep over many Tc values for the scatter plot
Tc_values = np.array([0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0])
num_trials = 5  # Multiple trials per Tc to get spread in the scatter plot

np.random.seed(42)

all_Tc = []
all_p01 = []
all_p10 = []
all_switching_time_01 = []  # Expected time in state 0 before switching: 1/p01
all_switching_time_10 = []  # Expected time in state 1 before switching: 1/p10

example_X = sample_gp(t_slots, sigma2, 2.0)

example_B = (example_X < S).astype(int)
print(f"First 100 bits: {''.join(map(str, example_B[:100]))}")
print()


for Tc in Tc_values:
    for trial in range(num_trials):
        X = sample_gp(t_slots, sigma2, Tc)

        B = (X < S).astype(int)

        # Count transitions
        n00 = n01 = n10 = n11 = 0
        for i in range(len(B) - 1):
            if B[i] == 0 and B[i+1] == 0:
                n00 += 1
            elif B[i] == 0 and B[i+1] == 1:
                n01 += 1
            elif B[i] == 1 and B[i+1] == 0:
                n10 += 1
            else:
                n11 += 1

        # Estimate transition probabilities
        p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        p10 = n10 / (n10 + n11) if (n10 + n11) > 0 else 0

        # Expected switching times (expected sojourn time in each state)
        sw_01 = 1.0 / p01 if p01 > 0 else np.nan
        sw_10 = 1.0 / p10 if p10 > 0 else np.nan

        all_Tc.append(Tc)
        all_p01.append(p01)
        all_p10.append(p10)
        all_switching_time_01.append(sw_01)
        all_switching_time_10.append(sw_10)

all_Tc = np.array(all_Tc)
all_p01 = np.array(all_p01)
all_p10 = np.array(all_p10)
all_switching_time_01 = np.array(all_switching_time_01)
all_switching_time_10 = np.array(all_switching_time_10)

# Average expected switching time (across both states)
all_avg_switching = (all_switching_time_01 + all_switching_time_10) / 2.0

print(f"{'Tc':>6}  {'p01':>8}  {'p10':>8}  {'E[T_0→1]':>10}  {'E[T_1→0]':>10}  {'Avg Switch':>10}")
print("-" * 65)
for i in range(len(all_Tc)):
    if i % num_trials == 0:  # Print first trial of each Tc
        print(f"{all_Tc[i]:6.1f}  {all_p01[i]:8.4f}  {all_p10[i]:8.4f}  "
              f"{all_switching_time_01[i]:10.2f}  {all_switching_time_10[i]:10.2f}  "
              f"{all_avg_switching[i]:10.2f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Expected switching time from state 0 (1/p01)
axes[0].scatter(all_Tc, all_switching_time_01, alpha=0.7, edgecolors='k', linewidths=0.5)
axes[0].set_xlabel('Coherence Time $T_c$', fontsize=12)
axes[0].set_ylabel('Expected Switching Time (slots)', fontsize=12)
axes[0].set_title('$1/p_{01}$: Expected time in state 0\nbefore switching to state 1', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Plot 2: Expected switching time from state 1 (1/p10)
axes[1].scatter(all_Tc, all_switching_time_10, alpha=0.7, color='orange', edgecolors='k', linewidths=0.5)
axes[1].set_xlabel('Coherence Time $T_c$', fontsize=12)
axes[1].set_ylabel('Expected Switching Time (slots)', fontsize=12)
axes[1].set_title('$1/p_{10}$: Expected time in state 1\nbefore switching to state 0', fontsize=12)
axes[1].grid(True, alpha=0.3)

# Plot 3: Average expected switching time
axes[2].scatter(all_Tc, all_avg_switching, alpha=0.7, color='green', edgecolors='k', linewidths=0.5)
axes[2].set_xlabel('Coherence Time $T_c$', fontsize=12)
axes[2].set_ylabel('Expected Switching Time (slots)', fontsize=12)
axes[2].set_title('Average Expected Switching Time\n$(1/p_{01} + 1/p_{10})/2$', fontsize=12)
axes[2].grid(True, alpha=0.3)

fig.suptitle(
    f'Gilbert-Elliott Channel: Expected Switching Time vs. Coherence Time\n'
    f'($\\sigma^2$={sigma2}, S={S}, D={D}, {num_slots} slots per sequence, {num_trials} trials per $T_c$)',
    fontsize=13, y=1.05
)
plt.tight_layout()
plt.savefig('/Users/qim/Desktop/EE597/assn 3/q2_scatter_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to q2_scatter_plot.png")
