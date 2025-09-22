**Smart Water Management in VIT Chennai using Causal Reinforcement Learning**

**Project Overview**

This project implements a smart water management system for the VIT Chennai campus using a Causal Reinforcement Learning (CRL) approach. The goal is to optimize water distribution, minimize wastage, and provide actionable insights for campus administrators. By analyzing historical water consumption data and simulating the effects of different interventions (like pipe repairs or water reallocation), the system learns an optimal policy for managing water resources efficiently.

This project moves beyond traditional prediction models by incorporating causality, allowing the RL agent to understand the real-world impact of its decisions.

**Project Structure**

The project is organized into the following Python scripts:

_requirements.txt_: A list of all the Python libraries required to run the project.

_data_preprocessor.py_: This script is responsible for cleaning and preparing the raw water consumption data (vit_chennai_water_dataset_noisy.csv). It handles missing values, corrects data types, and engineers relevant features for the CRL model.

_crl_water_management_model.py_: This is the core of the project. It defines the custom Reinforcement Learning environment for water management, including states, actions, and rewards. It also contains the implementation of the CRL agent which learns the optimal water management policy.

_main.py_: The main script that orchestrates the entire workflow. It loads the dataset, preprocesses the data using data_preprocessor.py, initializes and trains the CRL model from crl_water_management_model.py, and simulates the learned policy to demonstrate its effectiveness.

**Getting Started**

**Prerequisites**

Python 3.8 or higher

pip for installing Python packages

**Installation**

Clone the repository or download the project files.

Create and activate a virtual environment (recommended):

`python -m venv venv`
`venv\Scripts\activate`

Install the required libraries:

`pip install -r requirements.txt`

**How to Run the Project**

Make sure the vit_chennai_water_dataset_noisy.csv file is in the same directory as the Python scripts.

Execute the main script:

`python main.py`

The script will output the results of the simulation, showing the performance of the trained CRL agent in managing the campus's water resources.

**The Causal Reinforcement Learning Model**

**Environment**

State: The state is a representation of the current water situation on campus. It includes features like daily water usage, rainfall, temperature, humidity, pipe conditions, and special events.

Action: The agent can take several actions to manage water, such as dispatching a repair team, reallocating water between facilities, or issuing conservation alerts.

Reward: The reward function is designed to encourage efficient water management. The agent is rewarded for minimizing water wastage and penalized for shortages or inefficient allocation.

Agent
We use a Q-learning based agent that learns a policy to maximize the cumulative reward over time. By interacting with the simulated environment, the agent learns the causal effects of its actions on water consumption and availability.

Potential for Future Work
Integration with a live data stream for real-time water monitoring and management.

Development of a dashboard for campus administrators to visualize data and interact with the system's recommendations.

Expanding the action space of the agent to include more complex interventions.
