from data_preprocessor import preprocess_data
from crl_model import WaterManagementEnv, QLearningAgent, train_agent
import numpy as np

def run_simulation(env, agent):
    """
    Runs a simulation of the water management system with the trained agent.
    """
    print("\n--- Running Simulation with Trained Agent ---")
    state = env.reset()
    done = False
    total_reward = 0
    actions_taken = []
    action_map = {0: "Do nothing", 1: "Dispatch repair team", 2: "Reallocate water", 3: "Issue conservation alert"}

    step = 0
    while not done:
        # For simulation, we want to exploit the learned policy, so set epsilon to 0
        agent.epsilon = 0
        action = agent.choose_action(state) 
        actions_taken.append(action_map[action])
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        step +=1
        if done:
            print(f"Simulation finished after {step} steps.")
            break

    print(f"Total reward in simulation: {total_reward:.2f}")
    print("\nActions taken during simulation (first 40 days):")
    # Print first 20 actions for brevity
    for i, action in enumerate(actions_taken[:40]):
        print(f"Day {i+1}: {action}")


if __name__ == '__main__':
    # 1. Preprocess the data and select features for the RL state
    data_filepath = 'data/vit_chennai_water_dataset.csv'
    preprocessed_df = preprocess_data(data_filepath)

    # 2. Initialize the environment
    env = WaterManagementEnv(preprocessed_df)

    # 3. Initialize the agent
    # We pass the state data (without the target) to the agent for dynamic binning
    agent = QLearningAgent(
        state_data=env.state_space,
        action_space_n=env.action_space_n
    )
    
    # 4. Train the agent
    print("\n--- Training the Causal RL Agent ---")
    trained_agent = train_agent(env, agent, episodes=1000)


    # 5. Run a simulation with the trained agent
    run_simulation(env, trained_agent)

