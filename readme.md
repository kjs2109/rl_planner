# RL Path Planner 

### Agent Training & CARLA Simulator Evaluation
<table>
  <tr>
    <td valign="top">
      <h4 style="padding-bottom: 20px;">Training Environment</h4>
      <div>
        <img src="https://github.com/user-attachments/assets/291c4661-3dc6-4234-8a3e-41471ef8b956" width="280">
      </div>
    </td>
    <td valign="top">
      <h4>Illegally Parked Vehicle Scenario</h4>
      <video src="https://github.com/user-attachments/assets/27b09c1a-b79b-494a-a4c1-7b91ba153d57" width="450" controls></video>
    </td>
  </tr>
</table>

### Real-world Vehicle Validation
<table>
  <tr>
    <td valign="top">
      <ul>
        <li>도로유형: 1, 시나리오: 1</li>
      </ul>
      <video src="https://github.com/user-attachments/assets/b31641c5-26be-46b5-9aec-cdf5ccccd4ea" controls></video>
    </td>
    <td valign="top">
      <ul>
        <li>도로유형: 3, 시나리오: 1</li>
      </ul>
      <video src="https://github.com/user-attachments/assets/82f3836f-80f1-4ecf-a3d8-7d65839a636d" controls></video>
    </td>
  </tr>
</table>

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
> M. Jiang, Y. Li, S. Zhang, S. Chen, C. Wang, and M. Yang.  
> HOPE: A Reinforcement Learning-based Hybrid Policy Path Planner for Diverse Parking Scenarios.  
> arXiv preprint arXiv:2405.20579, 2024.
