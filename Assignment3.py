import random, math
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict

## Tic-tac-toe environment
player_X = 1   # opponent (random)
player_O = -1  # our MCTS agent


class TicTacToeState:
    """
    States: board of length 9 with entries in +1=X, -1=O, 0=empty and current player whose turn it is
    """
    def __init__(self, board: Optional[List[int]] = None, current_player: int = player_X):
        # if no board is igven, start with empty board
        self.board = board[:] if board is not None else [0] * 9
        self.current_player = current_player

    def copy(self):
        return TicTacToeState(self.board, self.current_player)

    def empty_cells(self) -> List[int]:
        # return list of available positions: A(x)
        return [i for i, v in enumerate(self.board) if v == 0]

    def play(self, action: int) -> "TicTacToeState":
        # apply action and switch player: return next state x'
        assert 0 <= action < 9 and self.board[action] == 0
        new_board = self.board[:]
        new_board[action] = self.current_player
        # flipping sign to switch to other player
        return TicTacToeState(new_board, -self.current_player)

    def lines(self):
        # all winning combinations (rows, cols and diagonals)
        return [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]

    def is_terminal(self) -> Tuple[bool, int]:
        """
        Check if the game is over.
        """
        # check all win combinations for three equal non-zero marks
        for a, b, c in self.lines():
            line_sum = self.board[a] + self.board[b] + self.board[c]
            if line_sum == 3 * player_X:
                return True, player_X
            if line_sum == 3 * player_O:
                return True, player_O
        # no winner: check for draw
        if all(v != 0 for v in self.board):
            return True, 0  # draw
        return False, 0    # game not finished

    def reward_for_O(self) -> Optional[float]:
        """
        Reward from O's perspective. only defined in terminal states
        - 1 if O win, 
        - 0 if X wins
        - 0.5 if draw
        """
        done, winner = self.is_terminal()
        if not done:
            return None
        if winner == player_O:
            return 1.0
        elif winner == player_X:
            return 0.0
        else:  # draw
            return 0.5

    def pretty(self) -> str:
        # 3x3 representation of the board
        symbols = {player_X: "X", player_O: "O", 0: "."}
        rows = []
        for r in range(3):
            rows.append(" ".join(symbols[self.board[3 * r + c]] for c in range(3)))
        return "\n".join(rows)


## MCTS tree node
class Node:
    def __init__(self, state: TicTacToeState,
                 parent: Optional["Node"] = None,
                 action_taken: Optional[int] = None):
        self.state = state # tic tac toe state at this node
        self.parent = parent # parent node
        self.action_taken = action_taken  # action from parent to reach this node
        self.children: Dict[int, Node] = {}  # action -> child node
        self.N = 0       # visit count
        self.W = 0.0     # total reward (for O)

    def is_fully_expanded(self) -> bool:
        # check if all available actions from this state have corresponsing child nodes.
        return len(self.children) == len(self.state.empty_cells())

    def best_child(self, c: float) -> "Node":
        # choose child with highest UCT
        assert self.children
        log_N = math.log(self.N)
        best, best_score = None, -1e9
        for a, child in self.children.items():
            exploit = child.W / child.N
            explore = c * math.sqrt(log_N / child.N)
            score = exploit + explore # UCT score
            if score > best_score:
                best, best_score = child, score
        return best



## MCTS: selection, expansion, simulation, backpropagation
def tree_policy(node: Node, c: float) -> Node:
    # Selection + expansion:
    # starting from node, descend tree using UCT until non-terminal 
    # node thats not fully expanded, then expand one child
   
    state = node.state
    while True:
        done, _ = state.is_terminal()
        if done:
            return node
        if not node.is_fully_expanded():
            # expansion: pick an unexpanded action from empty cells
            untried = [a for a in state.empty_cells() if a not in node.children]
            a = random.choice(untried)
            next_state = state.play(a)
            child = Node(next_state, parent=node, action_taken=a)
            node.children[a] = child
            return child
        else:
            # Selection: move to best child according to UCT
            node = node.best_child(c)
            state = node.state


def default_policy(state: TicTacToeState) -> float:
    # Simulation / rollout:
    # starting from state, play random moves for both players until terminal state
    # return terminal reward from O's perspective
    done, _ = state.is_terminal()
    while not done:
        actions = state.empty_cells()
        a = random.choice(actions)
        state = state.play(a)
        done, _ = state.is_terminal()
    reward = state.reward_for_O()
    assert reward is not None
    return reward


def backup(node: Node, reward: float):
    # Backpropagation: traverse path from leaf to root, update (N,W) every node on path

    while node is not None:
        node.N += 1
        node.W += reward
        node = node.parent


def mcts(root_state: TicTacToeState,
         num_simulations: int = 500,
         c: float = math.sqrt(2.0)) -> Tuple[int, Node]:
    
    # run MCTS from root for num simulations
    # return: best action with highest visit count and root node
    root = Node(root_state)

    for _ in range(num_simulations):
        leaf = tree_policy(root, c)
        reward = default_policy(leaf.state)
        backup(leaf, reward)

    # choose action with highest visit count
    if not root.children:
        return -1, root  # no empty cells
    best_action, best_child = max(root.children.items(), key=lambda kv: kv[1].N)
    return best_action, root


## Games and evaluations

# One game of tic tac toe
def play_one_game(num_simulations_per_move: int = 500, verbose: bool = True) -> Tuple[float, int]:

    # starter = X
    state = TicTacToeState(current_player=player_X)

    if verbose:
        print("Initial state:\n", state.pretty())
        print()

    while True:
        done, winner = state.is_terminal()
        if done:
            if verbose:
                print("Final state:\n", state.pretty())
                if winner == player_X:
                    print("X (random) wins")
                elif winner == player_O:
                    print("O (MCTS) wins")
                else:
                    print("Draw")
            reward = state.reward_for_O()
            if reward is None:
                reward = 0.0
            return reward, winner

        if state.current_player == player_X:
            # X uses random policy
            a = random.choice(state.empty_cells())
            state = state.play(a)
            if verbose:
                print("X plays", a)
                print(state.pretty())
                print()
        else:
            # O uses MCTS from current state
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
    # Evaluate MCTS agent against random X
    wins = draws = losses = 0

    for i in range(num_games):
        reward, winner = play_one_game(num_simulations_per_move=sims_per_move, verbose=False)
        if winner == player_O:
            wins += 1
        elif winner == player_X:
            losses += 1
        else:
            draws += 1

    print(f"Games: {num_games}, sims_per_move={sims_per_move}")
    print(f"O wins : {wins} ({wins/num_games:.3f})")
    print(f"Draws  : {draws} ({draws/num_games:.3f})")
    print(f"O loses: {losses} ({losses/num_games:.3f})")


## Convergence and action values

def diagnose_root_state(state: TicTacToeState,
                        simulation_steps: List[int],
                        c: float = math.sqrt(2.0)):
   
    # For fixed root state, run MCTS for different sim budgets, 
    # collect estimated Q values per available action
    results = []

    for n_sim in simulation_steps:
        action, root = mcts(state, num_simulations=n_sim, c=c)
        q_per_action = {}
        for a in state.empty_cells():
            child = root.children.get(a)
            if child is not None and child.N > 0:
                q_per_action[a] = child.W / child.N
            else:
                q_per_action[a] = None  # action never visited
        results.append((n_sim, q_per_action))

    return results

def print_q_values_as_grid(state: TicTacToeState, q_vals: Dict[int, Optional[float]]):
    
    # Print board and q value estimates for empty cells
    symbols = {player_X: "X", player_O: "O", 0: "."}
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
    
    state = TicTacToeState(current_player=player_X)
    history = []

    while True:
        done, winner = state.is_terminal()
        if done:
            reward = state.reward_for_O() or 0.0
            return reward, winner, history

        if state.current_player == player_X:
            # X: random
            a = random.choice(state.empty_cells())
            state = state.play(a)
        else:
            # O: MCTS, but keep root for logging q values
            a, root = mcts(state, num_simulations=num_simulations_per_move)

            # collect  qvalues per available action at this decision point
            q_vals = {}
            for act in state.empty_cells():
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
    # print each O decision with its estimated action values.
    
    print("Logged O-moves with estimated winning probabilities:\n")
    for move_idx, info in enumerate(history, start=1):
        board = info["board"]
        q_vals = info["q_vals"]
        print(f"--- O move {move_idx} ---")
        tmp_state = TicTacToeState(board=board, current_player=player_O)
        print_q_values_as_grid(tmp_state, q_vals)


## Plots

def make_convergence_plot():
    # State: X plays centre, O to move
    state = TicTacToeState()
    state = state.play(4)  # X in middle

    sims_list = [10, 50, 100, 200, 500, 1000]
    diag_results = diagnose_root_state(state, sims_list)

    # choose a few actions to track, e.g. corner points
    tracked_actions = [0, 2, 6, 8]

    for a in tracked_actions:
        y_vals = []
        for n_sim, q_vals in diag_results:
            q = q_vals.get(a)
            # if never visited, skip or set NaN
            y_vals.append(float('nan') if q is None else q)
        plt.plot(sims_list, y_vals, marker='o', label=f'action {a}')

    plt.xlabel('Simulations per move')
    plt.ylabel('Estimated win probability for O')
    plt.title('Convergence of MCTS value estimates (X in centre, O to move)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('convergence_qvalues.png', dpi=300)
    plt.close()


def plot_board_with_q(board, q_vals, title, filename):
    fig, ax = plt.subplots()

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')

    # no ticks / labels
    ax.set_xticks([])
    ax.set_yticks([])

    # tic-tac-toe grid
    for x in [0.5, 1.5]:
        ax.axvline(x, color="black", linewidth=2)
    for y in [0.5, 1.5]:
        ax.axhline(y, color="black", linewidth=2)

    # row 0 at top
    ax.invert_yaxis()

    # symbols and qvalues
    for r in range(3):
        for c in range(3):
            idx = 3 * r + c
            val = board[idx]

            if val == player_X:
                # red x center of cell
                ax.text(c, r, "X", ha="center", va="center",
                        fontsize=28, color="tab:red")
            elif val == player_O:
                # blue o center of cell
                ax.text(c, r, "O", ha="center", va="center",
                        fontsize=28, color="tab:blue")
            else:
                # empty cell: draw qvalue if available
                q = q_vals.get(idx)
                if q is not None:
                    ax.text(c, r, f"{q:.2f}", ha="center", va="center",
                            fontsize=12, color="dimgray")

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def make_logged_game_plots():
    reward, winner, history = play_one_game_with_logging(num_simulations_per_move=500)

    for i, info in enumerate(history[:3], start=1):
        board = info["board"]
        q_vals = info["q_vals"]
        title = f"O move {i}"
        filename = f"logged_game_move{i}.png"
        plot_board_with_q(board, q_vals, title, filename)


if __name__ == "__main__":
    random.seed(0)

    # single game demo
    r, w = play_one_game(num_simulations_per_move=200, verbose=True)
    print("Reward for O:", r, "winner:", w)

    # evaluation vs random X
    evaluate(num_games=200, sims_per_move=50)
    evaluate(num_games=200, sims_per_move=200)
    evaluate(num_games=200, sims_per_move=800)

    # convergence experiment from a fixed state
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

    # game plots
    make_logged_game_plots()

    # convergence plot
    make_convergence_plot()

