import numpy as np
import random

# Define the environment as a grid
board = [
    ['A', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
    ['D', 'W', 'W', 'W', 'D', 'W', 'W', 'W', 'D'],
    ['D', 'W', 'D', 'D', 'D', 'D', 'D', 'W', 'D'],
    ['D', 'D', 'D', 'W', 'E', 'W', 'D', 'D', 'D'],
    ['D', 'W', 'D', 'W', 'G', 'W', 'D', 'W', 'D'],
    ['D', 'W', 'D', 'D', 'W', 'D', 'D', 'W', 'D'],
    ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']
]

# Define actions and their corresponding indices
actions = {
    'up': 0,
    'down': 1,
    'left': 2,
    'right': 3
}

# Define rewards for actions
rewards = {
    1: 1,  # Dot collected
    -1: -1,  # Wall or ghost
    10: 10  # Goal state reached
}

# Initialize Q-Table with zeros
q_table = np.zeros((len(board), len(board[0]), len(actions)))

# Define epsilon-greedy policy parameters
epsilon = 0.1
alpha = 0.1
gamma = 0.9

# Convert the board into a numpy array
board = np.array(board)

# Define helper functions
def get_possible_actions(state):
    """Get possible actions from a given state."""
    possible_actions = []
    for action, idx in actions.items():
        new_state = get_new_state(state, idx)
        if is_valid_move(new_state):
            possible_actions.append(idx)
    return possible_actions

def get_new_state(state, action_idx):
    """Get the new state after taking an action."""
    action = list(actions.keys())[action_idx]
    if action == 'up':
        return (state[0] - 1, state[1])
    elif action == 'down':
        return (state[0] + 1, state[1])
    elif action == 'left':
        return (state[0], state[1] - 1)
    elif action == 'right':
        return (state[0], state[1] + 1)

def is_valid_move(state):
    """Check if a move is valid."""
    x, y = state
    return 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x, y] != 'W'

def choose_action(state):
    """Choose an action using epsilon-greedy policy."""
    if np.random.rand() < epsilon:
        return random.choice(get_possible_actions(state))
    else:
        return np.argmax(q_table[state[0], state[1]])

# Main loop
start_state = (0, 0)  # Starting state
current_state = start_state
goal_state = (3, 4)  # Define the goal state
while 'D' in board.flatten():
    action_idx = choose_action(current_state)
    action = list(actions.keys())[action_idx]
    new_state = get_new_state(current_state, action_idx)
    if is_valid_move(new_state):
        reward = rewards[1] if board[new_state] == 'D' else rewards[-1]
        q_table[current_state[0], current_state[1], action_idx] += alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[current_state[0], current_state[1], action_idx])
        board[current_state] = 'E'  # Mark current position as empty
        board[new_state] = 'A'  # Move the agent to the new position
        current_state = new_state

        # Display the board
        for row in board:
            print(' '.join(row))
        print()

        input("Press Enter to continue...")
    else:
        print("Invalid move. Try another action.")

print("No more dots left. Goal state reached!")
