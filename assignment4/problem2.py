import numpy as np
import matplotlib.pyplot as plt

g11, g22 = 1.0, 1.0
g12, g21 = 0.3, 0.3
N = 0.1
theta = 1.5

# FM update: each node sets power to exactly meet SINR threshold
def fm_update(P1, P2):
    P1_new = theta * (g21 * P2 + N) / g11
    P2_new = theta * (g12 * P1_new + N) / g22
    return P1_new, P2_new

# Run round-robin FM from an initial point
P1, P2 = 2.0, 0.1
traj = [(P1, P2)]
for _ in range(30):
    P1, P2 = fm_update(P1, P2)
    traj.append((P1, P2))
    if abs(traj[-1][0] - traj[-2][0]) < 1e-9:
        break
traj = np.array(traj)

# Minimum power solution (intersection)
# P1* = theta*(g21*P2*+N)/g11, P2* = theta*(g12*P1*+N)/g22
# Solve: symmetric case gives P* = theta*N/g11 / (1 - theta^2*g12*g21/(g11*g22))
a = theta * g21 / g11
b = theta * N / g11
c = theta * g12 / g22
d = theta * N / g22
P1_star = (b + a * d) / (1 - a * c)
P2_star = c * P1_star + d

# Plot
fig, ax = plt.subplots(figsize=(7, 6))

P_max = traj[:, 0].max() * 1.15
P1_range = np.linspace(0, P_max, 300)

# SINR1 = theta line: P1 = a*P2 + b  →  P2 = (P1 - b) / a
P2_line1 = (P1_range - b) / a

# SINR2 = theta line: P2 = c*P1 + d
P2_line2 = c * P1_range + d

ax.plot(P1_range, P2_line1, 'b-', label=r'SINR$_1 = \theta$ (link 1 threshold)')
ax.plot(P1_range, P2_line2, 'r-', label=r'SINR$_2 = \theta$ (link 2 threshold)')

# FM trajectory with arrows
ax.plot(traj[:, 0], traj[:, 1], 'k-o', markersize=4, linewidth=1, zorder=3, label='FM trajectory')
for i in range(len(traj) - 1):
    dx = traj[i+1, 0] - traj[i, 0]
    dy = traj[i+1, 1] - traj[i, 1]
    ax.annotate('', xy=(traj[i, 0] + 0.6*dx, traj[i, 1] + 0.6*dy),
                xytext=(traj[i, 0] + 0.4*dx, traj[i, 1] + 0.4*dy),
                arrowprops=dict(arrowstyle='->', color='k', lw=1.2))

ax.scatter(*traj[0], s=100, color='green', zorder=5, label=f'Start ({traj[0,0]}, {traj[0,1]})')
ax.scatter(P1_star, P2_star, s=150, marker='*', color='gold', zorder=5,
           label=f'Min-power solution ({P1_star:.3f}, {P2_star:.3f})')

ax.set_xlim(0, P_max)
ax.set_ylim(0, traj[:, 1].max() * 1.15)
ax.set_xlabel('$P_1$', fontsize=13)
ax.set_ylabel('$P_2$', fontsize=13)
ax.set_title(
    f'Foschini-Miljanic Algorithm  '
    f'($g_{{11}}=g_{{22}}={g11}$, $g_{{12}}=g_{{21}}={g12}$, $N={N}$, $\\theta={theta}$)',
    fontsize=11
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assignment4/problem2.png', dpi=150, bbox_inches='tight')
print("Saved assignment4/problem2.png")
plt.show()
