# Smart Water Management in VIT Chennai using Causal Reinforcement Learning

## 📌 Project Overview
This project implements a **Smart Water Management System** for the **VIT Chennai campus** using a **Causal Reinforcement Learning (CRL)** approach.  

The goal is to:
- Optimize water distribution  
- Minimize wastage  
- Provide actionable insights for campus administrators  

Unlike traditional prediction models, this system **incorporates causality**, allowing the RL agent to understand the real-world impact of its decisions. It leverages historical water consumption data and simulates interventions (e.g., pipe repairs, water reallocation) to learn an **optimal water management policy**.

---

## 📂 Project Structure
The project is organized into the following Python scripts:

- **`requirements.txt`** → List of all required Python libraries.  
- **`data_preprocessor.py`** → Cleans and prepares the raw dataset (`data/vit_chennai_water_dataset.csv`), handles missing values, corrects data types, and engineers features for the CRL model.  
- **`crl_model.py`** → Defines the custom RL environment (states, actions, rewards) and implements the CRL agent that learns the optimal policy.  
- **`main_processor.py`** → Orchestrates the workflow:  
  - Loads dataset  
  - Preprocesses data  
  - Trains CRL model  
  - Simulates learned policy to evaluate performance  

---

## ⚙️ Getting Started

### ✅ Prerequisites
- Python **3.8+**
- `pip` for package management

### 🔧 Installation
1. Clone the repository or download the project files.  
2. Create and activate a virtual environment (recommended):  
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate   # On Linux/Mac
   ```
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run the Project
1. Ensure `vit_chennai_water_dataset.csv` is in the same directory as the scripts.  
2. Run the main script:  
   ```bash
   python main.py
   ```
3. The script outputs **simulation results**, showing the performance of the trained CRL agent in managing water resources.

---

## 🧠 The Causal Reinforcement Learning Model

### 🌍 Environment
- **State**: Daily water usage, rainfall, temperature, humidity, pipe conditions, special events.  
- **Action**: Repair pipes, reallocate water, issue conservation alerts, etc.  
- **Reward**: Rewards efficient management (minimized wastage) and penalizes shortages or poor allocation.  

### 🤖 Agent
- Based on **Q-Learning**.  
- Learns a **policy** to maximize cumulative rewards.  
- Understands **causal effects** of interventions on water availability and consumption.  

---

## 🔮 Future Work
- Integration with **live data streams** for real-time monitoring.  
- Development of a **dashboard** for administrators to visualize insights.  
- Expanding the **action space** to support more complex interventions.  

---

## 📜 License
This project is for **research and educational purposes**. Please cite appropriately if used.  
