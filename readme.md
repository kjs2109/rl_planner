# RL Path Planner 
### Agent Training
![Image](https://github.com/user-attachments/assets/291c4661-3dc6-4234-8a3e-41471ef8b956)

### Planning Simulator Evaluation (Integrated with the Autoware System)
![Image](https://github.com/user-attachments/assets/c975c846-61dd-49d1-87b0-5309b8e405a0)

### End-to-End Simulator Evaluation 
![Image](https://github.com/user-attachments/assets/850a66bd-a0f7-4c88-ab67-63e3222b88f0)

### Setup Environment 
```
python3 -m venv rl_env
source rl_env/bin/activate
pip install -r requirements.txt
colcon build --symlink-install --base-path ./ros/
source set_env.sh
```

### reference  
paper
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
- [HOPE: A Reinforcement Learning-based Hybrid Policy Path Planner for Diverse Parking Scenarios](https://arxiv.org/abs/2405.20579)  

code 
- https://github.com/jiamiya/HOPE
