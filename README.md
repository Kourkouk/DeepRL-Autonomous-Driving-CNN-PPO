# Autonomous Vehicle Navigation in Non-Deterministic Environments

## Overview
This project focuses on the development and training of an intelligent agent that learns to autonomously drive a vehicle in procedurally generated tracks, without any prior knowledge of the track's geometry or GPS coordinates. The simulation is built upon the continuous control environment `CarRacing-v3` from the Gymnasium library. 

The solution leverages a combination of Deep Learning (DL) for visual perception and Reinforcement Learning (RL) for decision making.

## Observation and Action Space
* **Observations:** The agent relies exclusively on raw optical information (raw pixels). It receives 96x96 RGB images from a top-down camera mounted above the vehicle. To provide the agent with short-term temporal memory (speed, direction, drift), a **Frame Stacking** technique of 4 consecutive frames is applied, resulting in a 96x96x12 input matrix.
* **Action Space:** The agent controls three continuous driving variables simultaneously: steering angle, acceleration (gas), and deceleration (brake).
* **Reward:** The agent is rewarded for successfully traversing new track segments and penalized for going off-road or lacking progress over time.

## Architecture
The autonomous driving system is structurally divided into two main subsystems:
1. **Perception (Vision) - Custom CNN:** A custom 3-layer Convolutional Neural Network acts as a feature extractor. It uses an aggressive downsampling strategy to compress the spatial and temporal information into a tiny 3x3 grid. This is then flattened and passed through a fully connected layer with 256 neurons, forcing the network to discard environmental noise and retain only essential driving features[cite: 98, 99].
2. **Decision Making (Brain) - PPO:** The extracted 256-feature vector is fed into the state-of-the-art **Proximal Policy Optimization (PPO)** algorithm. Built on an Actor-Critic architecture , the PPO was configured with strict clipping (`clip_range = 0.15`), an entropy coefficient (`ent_coef = 0.01`) to encourage exploration, and a linear learning rate schedule for optimal convergence.

## Training & Results
Through extensive grid search (evaluating learning rates, network depth, and feature dimensions) , the final model was trained for 500,000 timesteps. To prevent "reward hacking" (such as continuous drifting to avoid crashes) and policy collapse in the late training stages, an `EvalCallback` routine was introduced. 

[cite_start]This allowed the system to save the model at its absolute peak performance, achieving a high score of **~880**.

## Technologies & Libraries Used
* **Python** (IDE: PyCharm)
* **Gymnasium** (with `Box2D` for the CarRacing-v3 environment)
* **Stable-Baselines3** (for the optimized PPO implementation)
* **PyTorch** (for building and training the custom CNN)
* **TensorBoard** (for tracking metrics and visualizing training curves)

## Project Structure
* `test_model_PPO.py` / `test_agent_PPO.py`: Scripts used during the testing phase (Grid Search) for hyperparameter tuning.
* `Final_model/`: Directory containing the optimized final code.
  * `model_final_PPO.py`: The final optimal network architecture.
  * `agent_final_PPO.py`: Script for the final 500k-step training process.
  * `car_racing_final_PPO.py`: Rendering script to load the saved best model (`best_model.zip`) and watch the agent drive autonomously.
