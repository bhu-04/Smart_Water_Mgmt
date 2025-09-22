import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class WaterManagementEnv:
    """
    A custom environment for the Causal Reinforcement Learning agent
    that simulates water management at VIT Chennai.
    """
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.max_steps = len(data) - 1

        # Define the state space (all columns except our target 'water_stress_level')
        self.state_space = self.data.drop('water_stress_level', axis=1)
        self.observation_space_dim = len(self.state_space.columns)

        # Define the action space
        # 0: Do nothing
        # 1: Dispatch repair team
        # 2: Reallocate water
        # 3: Issue conservation alert
        self.action_space_n = 4

    def reset(self):
        """Resets the environment to the initial state."""
        self.current_step = 0
        return self.state_space.iloc[self.current_step].values

    def step(self, action):
        """
        Takes an action and returns the next state, reward, and whether the episode is done.
        """
        if self.current_step >= self.max_steps:
            # Episode is done
            return None, 0, True

        current_state_row = self.data.iloc[self.current_step]
        reward = self._calculate_reward(current_state_row, action)

        self.current_step += 1
        next_state = self.state_space.iloc[self.current_step].values

        done = self.current_step >= self.max_steps
        return next_state, reward, done

    def _calculate_reward(self, state_row, action):
        """
        Calculates the reward based on the current state and the action taken.
        This is where the "causal" aspect is modeled implicitly.
        """
        reward = 0
        # Base reward for maintaining a good state
        if state_row['water_stress_level'] == 0:
            reward += 5

        # Penalize for high water stress
        if state_row['water_stress_level'] > 1:
            reward -= 10 * state_row['water_stress_level']

        # Reward for taking appropriate actions
        if action == 1 and state_row['Leak_Reports'] > 0: # Dispatch repair team for leaks
            reward += 20
        elif action == 1 and state_row['Leak_Reports'] == 0: # Penalize unnecessary action
            reward -= 5

        if action == 3 and state_row['water_stress_level'] > 2: # Issue alert during high stress
            reward += 15
        elif action == 3 and state_row['water_stress_level'] <= 1:
             reward -=5


        return reward

class QLearningAgent:
    """
    A Q-learning agent that learns an optimal policy for water management.
    """
    def __init__(self, observation_space_dim, action_space_n, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay_rate=0.995, min_exploration_rate=0.01):
        # The Q-table now needs to handle continuous states by discretization
        self.state_bins = [np.linspace(0, 500000, 10) for _ in range(observation_space_dim)] # Example bins
        self.q_table = np.zeros([10] * observation_space_dim + [action_space_n])


        self.action_space_n = action_space_n
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay_rate
        self.epsilon_min = min_exploration_rate

    def _discretize_state(self, state):
        """Discretizes a continuous state into a tuple of bin indices."""
        binned_state = []
        # Ensure state is a 1D array
        state = np.atleast_1d(state)
        # Check if the number of elements in state matches the number of bins
        if len(state) != len(self.state_bins):
            # This is a failsafe. You should debug why the state dimension is incorrect.
            # For now, we'll pad or truncate. Padding is safer.
            new_state = np.zeros(len(self.state_bins))
            size = min(len(state), len(self.state_bins))
            new_state[:size] = state[:size]
            state = new_state

        for i, val in enumerate(state):
             binned_state.append(np.digitize(val, self.state_bins[i]) - 1)
        return tuple(binned_state)


    def choose_action(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space_n)  # Explore
        else:
            discrete_state = self._discretize_state(state)
            return np.argmax(self.q_table[discrete_state])  # Exploit

    def learn(self, state, action, reward, next_state):
        """Updates the Q-table using the Q-learning formula."""
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)

        old_value = self.q_table[discrete_state + (action,)]
        next_max = np.max(self.q_table[discrete_next_state])

        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[discrete_state + (action,)] = new_value

    def update_exploration_rate(self):
        """Decays the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_agent(env, agent, episodes=1000):
    """
    Trains the Q-learning agent.
    """
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            if not done:
                agent.learn(state, action, reward, next_state)
                state = next_state
                episode_reward += reward

        agent.update_exploration_rate()
        total_rewards.append(episode_reward)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {episode_reward}")

    print("\nTraining complete.")
    return agent
