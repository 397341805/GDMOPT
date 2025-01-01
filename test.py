import numpy as np
import gymnasium as gym

class DiffusionQLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((state_space, action_space))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def diffuse_q_values(self):
        # Applying diffusion to the Q-table for smoother exploration
        for state in range(self.state_space):
            for action in range(self.action_space):
                neighbors = self.get_neighbors(state)
                diffusion_value = np.mean([self.q_table[neighbor][action] for neighbor in neighbors])
                self.q_table[state][action] = 0.5 * self.q_table[state][action] + 0.5 * diffusion_value

    def get_neighbors(self, state):
        # Define neighbors for diffusion, assuming a 1D state space for simplicity
        neighbors = []
        if state > 0:
            neighbors.append(state - 1)
        if state < self.state_space - 1:
            neighbors.append(state + 1)
        return neighbors

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)
    state_space = env.observation_space.n
    action_space = env.action_space.n

    agent = DiffusionQLearningAgent(state_space, action_space)
    episodes = 1000
    max_steps = 100

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.update_q_value(state, action, reward, next_state)
            agent.diffuse_q_values()

            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    print("Training finished.")
