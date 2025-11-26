import itertools

# Call attempts per minute in each cell
lambdas = [2.0, 5.0, 8.0, 9.0, 11.0]  

beta = 1.5  # mean call duration (minutes)
rhos = [lam * beta for lam in lambdas]  

total_lambda = sum(lambdas)
ps = [lam / total_lambda for lam in lambdas]  # p_i from Q10

NUM_CELLS = 5

neighbour_pairs = {
    (0, 1), (1, 0),  # 1–2
    (0, 2), (2, 0),  # 1–3
    (1, 2), (2, 1),  # 2–3
    (2, 3), (3, 2),  # 3–4
    (2, 4), (4, 2),  # 3–5
    (3, 4), (4, 3),  # 4–5
}

def is_independent_set(cells_subset):
    # Given a set of cells it checks if its alloud to use one channel 
    # in all these cells at the same time. 
    cells_list = list(cells_subset)
    for i in range(len(cells_list)):
        for j in range(i + 1, len(cells_list)):
            a, b = cells_list[i], cells_list[j]
            if (a, b) in neighbour_pairs:
                return False
    return True

def erlang_b(rho, N):
    if N == 0:
        return 1.0
    B = 1.0
    for n in range(1, N + 1):
        B = (rho * B) / (n + rho * B)
    return B

def compute_blocking(Ns):
    B_cells = [erlang_b(rhos[i], Ns[i]) for i in range(NUM_CELLS)]
    B_overall = sum(ps[i] * B_cells[i] for i in range(NUM_CELLS))
    return B_overall, B_cells

all_cells = list(range(NUM_CELLS))
allowed_patterns = []
for r in range(1, NUM_CELLS + 1):
    for subset in itertools.combinations(all_cells, r): # generates all combinations
        if is_independent_set(subset):
            allowed_patterns.append(tuple(subset))

def greedy_channel_allocation(NUM_CHANNELS):
    Ns = [0] * NUM_CELLS 

    pattern_counts = {p: 0 for p in allowed_patterns} # dict of legal patterns and how much we use the pattern
    channel_assignments = {} # will store for each channel to which pattern it's assigned to

    for ch in range(1, NUM_CHANNELS + 1):
        best_pattern = None
        best_B_new = None
        best_Ns = None

        for pattern in allowed_patterns:
            candidate_Ns = Ns.copy()
            for i in pattern:
                candidate_Ns[i] += 1
            B_new, _ = compute_blocking(candidate_Ns)

            # Keep the best pattern
            if (best_B_new is None) or (B_new < best_B_new - 1e-12):
                best_B_new = B_new
                best_pattern = pattern
                best_Ns = candidate_Ns

        Ns = best_Ns
        pattern_counts[best_pattern] += 1
        channel_assignments[ch] = best_pattern

    B_overall, B_cells = compute_blocking(Ns)
    return Ns, B_cells, B_overall, pattern_counts, channel_assignments

def find_min_channels(target=0.01, C_start=32, C_max=120):
    for C in range(C_start, C_max + 1):
        Ns, B_cells, B_overall, pattern_counts, channel_assignments = \
            greedy_channel_allocation(C)

        print(f"C = {C:3d}: B_overall = {B_overall:.4f}")

        if B_overall < target:
            return C, Ns, B_cells, B_overall, pattern_counts, channel_assignments

    # If nothing found up to C_max
    return None, None, None, None, None, None

if __name__ == "__main__":
    # QUESTION 11

    NUM_CHANNELS = 32
    Ns, B_cells, B_overall, pattern_counts, channel_assignments = greedy_channel_allocation(NUM_CHANNELS)

    print("Per-cell channel counts N_i (cells 1..5):", Ns)
    print("Per-cell blocking probabilities B_i:", B_cells)
    print("Overall blocking probability:", B_overall)

    # QUESTION 12
    target = 0.01
    C_min, Ns, B_cells, B_overall, pattern_counts, channel_assignments = find_min_channels(target=target, C_start=32, C_max=120)

    print(f"Minimal total number of channels C*: {C_min}")
    print("Per-cell channel counts N_i (cells 1..5):", Ns)
    print("Per-cell blocking probabilities B_i:", B_cells)
    print("Overall blocking probability:", B_overall)
