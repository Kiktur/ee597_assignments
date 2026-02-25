import numpy as np
import matplotlib.pyplot as plt

def build_cov_matrix(t, sigma2, Tc):
    """Build the covariance matrix for time vector t."""
    diff = t[:, np.newaxis] - t[np.newaxis, :]
    K = sigma2 * np.exp(-(diff ** 2) / (Tc ** 2))
    return K

def sample_gp(t, sigma2, Tc, num_samples=3):
    """Sample from a zero-mean GP."""
    K = build_cov_matrix(t, sigma2, Tc)
    mean = np.zeros(len(t))
    samples = np.random.multivariate_normal(mean, K, size=num_samples)
    return samples

# Parameters
sigma2 = 1.0
Tc_values = [0.5, 2.0, 8.0]  # Three different coherence times
t = np.linspace(0, 20, 300)
num_samples = 3

np.random.seed(42)

fig, axes = plt.subplots(len(Tc_values), 1, figsize=(12, 10), sharex=True)

for idx, Tc in enumerate(Tc_values):
    samples = sample_gp(t, sigma2, Tc, num_samples)
    ax = axes[idx]
    for i in range(num_samples):
        ax.plot(t, samples[i], label=f'Sample {i+1}', alpha=0.8)
    ax.set_title(f'Gaussian Fading Process  —  $T_c$ = {Tc}', fontsize=13)
    ax.set_ylabel('X(t)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-4, 4)

axes[-1].set_xlabel('Time t')
fig.suptitle(
    r"Fading as Gaussian Process: $K(t,t') = \sigma^2 \exp\!\left(-(t-t')^2 / T_c^2\right)$"
    f"\n($\\sigma^2$ = {sigma2})",
    fontsize=14, y=1.02
)
plt.tight_layout()
plt.savefig('/Users/qim/Desktop/EE597/assn 3/q1_fading_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to q1_fading_plot.png")
