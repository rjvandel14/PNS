import random, math
from typing import List, Optional, Tuple, Dict

PLAYER_X = 1   # opponent (random)
PLAYER_O = -1  # our MCTS agent


class TicTacToeState:
    def __init__(self, board: Optional[List[int]] = None, current_player: int = PLAYER_X):
        self.board = board[:] if board is not None else [0] * 9
        self.current_player = current_player

    def copy(self):
        return TicTacToeState(self.board, self.current_player)

    def legal_actions(self) -> List[int]:
        """Return list of empty positions (0..8)."""
        return [i for i, v in enumerate(self.board) if v == 0]

    def play(self, action: int) -> "TicTacToeState":
        """Return next state after current_player plays at index `action`."""
        assert 0 <= action < 9 and self.board[action] == 0
        new_board = self.board[:]
        new_board[action] = self.current_player
        return TicTacToeState(new_board, -self.current_player)

    def lines(self):
        """All 8 win lines as triples of indices."""
        return [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]

    def is_terminal(self) -> Tuple[bool, int]:
        """
        Returns (done, winner), winner in {PLAYER_X, PLAYER_O, 0}
        0 means no winner (either draw or non-terminal).
        """
        for a, b, c in self.lines():
            line_sum = self.board[a] + self.board[b] + self.board[c]
            if line_sum == 3 * PLAYER_X:
                return True, PLAYER_X
            if line_sum == 3 * PLAYER_O:
                return True, PLAYER_O
        # no winner: check for draw
        if all(v != 0 for v in self.board):
            return True, 0  # draw
        return False, 0    # game not finished

    def reward_for_O(self) -> Optional[float]:
        """
        Reward from O's perspective.
        1 for O win, 0 for loss, 0 for draw (you can change draw to 0.5 if you want).
        """
        done, winner = self.is_terminal()
        if not done:
            return None
        if winner == PLAYER_O:
            return 1.0
        elif winner == PLAYER_X:
            return 0.0
        else:  # draw
            return 0.0

    def pretty(self) -> str:
        symbols = {PLAYER_X: "X", PLAYER_O: "O", 0: "."}
        rows = []
        for r in range(3):
            rows.append(" ".join(symbols[self.board[3 * r + c]] for c in range(3)))
        return "\n".join(rows)


class Node:
    def __init__(self, state: TicTacToeState,
                 parent: Optional["Node"] = None,
                 action_taken: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken  # action from parent to reach this node
        self.children: Dict[int, Node] = {}  # action -> child node
        self.N = 0       # visit count
        self.W = 0.0     # total reward (for O)

    def is_fully_expanded(self) -> bool:
        """All legal actions from this state have been expanded."""
        return len(self.children) == len(self.state.legal_actions())

    def best_child(self, c: float) -> "Node":
        """
        Select child with highest UCT score:
        Q + c * sqrt(log(N_parent) / N_child)
        """
        assert self.children
        log_N = math.log(self.N)
        best, best_score = None, -1e9
        for a, child in self.children.items():
            exploit = child.W / child.N
            explore = c * math.sqrt(log_N / child.N)
            score = exploit + explore
            if score > best_score:
                best, best_score = child, score
        return best


def tree_policy(node: Node, c: float) -> Node:
    """
    Selection + Expansion.
    Starting from `node`, descend the tree using UCT until we hit
    a non-terminal node that is not fully expanded, then expand one child.
    """
    state = node.state
    while True:
        done, _ = state.is_terminal()
        if done:
            return node
        if not node.is_fully_expanded():
            # Expansion: pick an untried action
            untried = [a for a in state.legal_actions() if a not in node.children]
            a = random.choice(untried)
            next_state = state.play(a)
            child = Node(next_state, parent=node, action_taken=a)
            node.children[a] = child
            return child
        else:
            # Selection: follow best child by UCT
            node = node.best_child(c)
            state = node.state


def default_policy(state: TicTacToeState) -> float:
    """
    Rollout (simulation): play random moves for both players until terminal.
    Return reward from O's perspective.
    """
    done, _ = state.is_terminal()
    while not done:
        actions = state.legal_actions()
        a = random.choice(actions)
        state = state.play(a)
        done, _ = state.is_terminal()
    reward = state.reward_for_O()
    assert reward is not None
    return reward


def backup(node: Node, reward: float):
    """
    Backpropagation: update N and W along the path from node to root.
    """
    while node is not None:
        node.N += 1
        node.W += reward
        node = node.parent


def mcts(root_state: TicTacToeState,
         num_simulations: int = 500,
         c: float = math.sqrt(2.0)) -> Tuple[int, Node]:
    """
    Run MCTS from `root_state` and return:
    - best_action according to visit count
    - root node (so you can inspect children stats for plots)
    """
    root = Node(root_state)

    for _ in range(num_simulations):
        leaf = tree_policy(root, c)
        reward = default_policy(leaf.state)
        backup(leaf, reward)

    # choose action with highest visit count
    if not root.children:
        return -1, root  # no legal moves
    best_action, best_child = max(root.children.items(), key=lambda kv: kv[1].N)
    return best_action, root


def play_one_game(num_simulations_per_move: int = 500, verbose: bool = True) -> Tuple[float, int]:
    """
    Play one game:
    - X: random policy
    - O: MCTS with `num_simulations_per_move`
    Returns (reward_for_O, winner), winner in {PLAYER_X, PLAYER_O, 0}.
    """
    state = TicTacToeState(current_player=PLAYER_X)

    if verbose:
        print("Initial state:\n", state.pretty())
        print()

    while True:
        done, winner = state.is_terminal()
        if done:
            if verbose:
                print("Final state:\n", state.pretty())
                if winner == PLAYER_X:
                    print("X (random) wins")
                elif winner == PLAYER_O:
                    print("O (MCTS) wins")
                else:
                    print("Draw")
            reward = state.reward_for_O()
            if reward is None:
                reward = 0.0
            return reward, winner

        if state.current_player == PLAYER_X:
            # X: random
            a = random.choice(state.legal_actions())
            state = state.play(a)
            if verbose:
                print("X plays", a)
                print(state.pretty())
                print()
        else:
            # O: MCTS
            a, root = mcts(state, num_simulations=num_simulations_per_move)
            if a == -1:
                # no moves possible
                done, winner = state.is_terminal()
                reward = state.reward_for_O() or 0.0
                return reward, winner

            if verbose:
                print("O thinking...")
                for action, child in root.children.items():
                    est = child.W / child.N
                    print(f"  action {action}: N={child.N}, Q={est:.3f}")

            state = state.play(a)
            if verbose:
                print("O plays", a)
                print(state.pretty())
                print()


def evaluate(num_games: int = 200, sims_per_move: int = 200) -> None:
    """
    Play `num_games` games vs random X and print win/draw/loss stats for O.
    """
    wins = draws = losses = 0

    for i in range(num_games):
        reward, winner = play_one_game(num_simulations_per_move=sims_per_move, verbose=False)
        if winner == PLAYER_O:
            wins += 1
        elif winner == PLAYER_X:
            losses += 1
        else:
            draws += 1

    print(f"Games: {num_games}, sims_per_move={sims_per_move}")
    print(f"O wins : {wins} ({wins/num_games:.3f})")
    print(f"Draws  : {draws} ({draws/num_games:.3f})")
    print(f"O loses: {losses} ({losses/num_games:.3f})")


def diagnose_root_state(state: TicTacToeState,
                        simulation_steps: List[int],
                        c: float = math.sqrt(2.0)):
    """
    For a fixed state, run MCTS multiple times with increasing num_simulations,
    and collect estimated Q-values per legal action at the root.

    Returns a list of (num_simulations, {action: Q}) pairs.
    """
    results = []

    for n_sim in simulation_steps:
        action, root = mcts(state, num_simulations=n_sim, c=c)
        q_per_action = {}
        for a in state.legal_actions():
            child = root.children.get(a)
            if child is not None and child.N > 0:
                q_per_action[a] = child.W / child.N
            else:
                q_per_action[a] = None  # never visited
        results.append((n_sim, q_per_action))

    return results

def print_q_values_as_grid(state: TicTacToeState, q_vals: Dict[int, Optional[float]]):
    """
    Print the board with Q-values in the empty cells.
    """
    symbols = {PLAYER_X: "X", PLAYER_O: "O", 0: "."}
    for r in range(3):
        row_cells = []
        for c in range(3):
            idx = 3 * r + c
            if state.board[idx] != 0:
                row_cells.append(f" {symbols[state.board[idx]]}   ")
            else:
                q = q_vals.get(idx)
                if q is None:
                    row_cells.append(" .   ")
                else:
                    row_cells.append(f"{q:0.2f}")
        print(" ".join(row_cells))
    print()


def play_one_game_with_logging(num_simulations_per_move: int = 200):
    """
    Like play_one_game, but log for each O-move:
    - the board state
    - the estimated Q-values per legal action at the root.

    Returns:
        history: list of dicts with keys:
            'board'  : list[int] of length 9
            'q_vals' : dict[action -> Q]
    """
    state = TicTacToeState(current_player=PLAYER_X)
    history = []

    while True:
        done, winner = state.is_terminal()
        if done:
            reward = state.reward_for_O() or 0.0
            return reward, winner, history

        if state.current_player == PLAYER_X:
            # X: random
            a = random.choice(state.legal_actions())
            state = state.play(a)
        else:
            # O: MCTS, but keep root for logging
            a, root = mcts(state, num_simulations=num_simulations_per_move)

            # collect Q-values per legal action at this decision point
            q_vals = {}
            for act in state.legal_actions():
                child = root.children.get(act)
                if child is not None and child.N > 0:
                    q_vals[act] = child.W / child.N
                else:
                    q_vals[act] = None

            history.append({
                "board": state.board[:],
                "q_vals": q_vals
            })

            state = state.play(a)

def show_logged_game(history):
    """
    Print each O decision with its estimated action values.
    """
    print("Logged O-moves with estimated winning probabilities:\n")
    for move_idx, info in enumerate(history, start=1):
        board = info["board"]
        q_vals = info["q_vals"]
        print(f"--- O move {move_idx} ---")
        tmp_state = TicTacToeState(board=board, current_player=PLAYER_O)
        print_q_values_as_grid(tmp_state, q_vals)




if __name__ == "__main__":
    random.seed(0)

    # Quick single game demo
    r, w = play_one_game(num_simulations_per_move=200, verbose=True)
    print("Reward for O:", r, "winner:", w)

    # Evaluation vs random X
    evaluate(num_games=200, sims_per_move=50)
    evaluate(num_games=200, sims_per_move=200)
    evaluate(num_games=200, sims_per_move=800)

    # 2) Convergence experiment from a fixed state
    start_state = TicTacToeState()
    # X plays center (index 4), now O to move
    start_state = start_state.play(4)

    sims_list = [10, 50, 100, 200, 500, 1000]
    diag_results = diagnose_root_state(start_state, sims_list)

    for n_sim, q_vals in diag_results:
        print(f"\n=== Simulations: {n_sim} ===")
        print_q_values_as_grid(start_state, q_vals)


    reward, winner, hist = play_one_game_with_logging(num_simulations_per_move=500)
    print("Final reward:", reward, "winner:", winner)
    show_logged_game(hist)

