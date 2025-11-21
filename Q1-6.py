
import numpy as np

# -----------------------------
# Parameters (from the assignment)
# -----------------------------
A = 1.2  # km^2

lambda_voice = 25 * A      # = 30 calls/hour
lambda_total_vc = 0.8 * A  # = 0.96 calls/hour
lambda_low = 0.8 * lambda_total_vc   # = 0.768
lambda_high = 0.2 * lambda_total_vc  # = 0.192

mu_voice = 12.0              # 5 minutes -> 12 per hour
mu_low = mu_high = 10.0 / 3  # 18 minutes -> 10/3 per hour

C = 4  # total channels
b_voice, b_low, b_high = 1, 2, 3  # channels per call

# -----------------------------
# State space S
# (n_voice, n_low, n_high)
# satisfying n_voice + 2 n_low + 3 n_high <= 4
# -----------------------------
states = [
    (0, 0, 0),
    (1, 0, 0),
    (2, 0, 0),
    (3, 0, 0),
    (4, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (2, 1, 0),
    (0, 2, 0),
    (0, 0, 1),
    (1, 0, 1),
]

n_states = len(states)
state_index = {s: idx for idx, s in enumerate(states)}

# -----------------------------
# Build generator matrix Q
# Q[i, j] = rate of transition i -> j
# -----------------------------
Q = np.zeros((n_states, n_states))

for idx, (nv, nl, nh) in enumerate(states):
    used = nv + 2 * nl + 3 * nh

    # Arrivals (if enough capacity)
    # Voice
    if used + b_voice <= C:
        s2 = (nv + 1, nl, nh)
        Q[idx, state_index[s2]] += lambda_voice

    # Low-res video
    if used + b_low <= C:
        s2 = (nv, nl + 1, nh)
        Q[idx, state_index[s2]] += lambda_low

    # High-res video
    if used + b_high <= C:
        s2 = (nv, nl, nh + 1)
        Q[idx, state_index[s2]] += lambda_high

    # Departures (if at least one call of that type)
    if nv > 0:
        s2 = (nv - 1, nl, nh)
        Q[idx, state_index[s2]] += nv * mu_voice

    if nl > 0:
        s2 = (nv, nl - 1, nh)
        Q[idx, state_index[s2]] += nl * mu_low

    if nh > 0:
        s2 = (nv, nl, nh - 1)
        Q[idx, state_index[s2]] += nh * mu_high

    # Diagonal entry so that each row sums to zero
    Q[idx, idx] = -Q[idx].sum()

# -----------------------------
# Solve for stationary distribution pi
# pi Q = 0, sum(pi) = 1
# -----------------------------
# We solve Q^T pi^T = 0 with one row replaced by the normalization condition.
A_mat = Q.T.copy()
b_vec = np.zeros(n_states)

# Replace last equation by sum(pi) = 1
A_mat[-1, :] = 1.0
b_vec[-1] = 1.0

pi = np.linalg.solve(A_mat, b_vec)

# Print results
for (nv, nl, nh), p in zip(states, pi):
    print(f"pi({nv},{nl},{nh}) = {p:.5f}")

print("Sum pi =", pi.sum())
