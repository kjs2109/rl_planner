import os 
import time
import json 
from shutil import copyfile 
import argparse 
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
from torch.utils.tensorboard import SummaryWriter 

from model.agent.sac_agent import SACAgent as SAC 
from model.agent.agent import PlanningAgent 
from env.vehicle import Status 
from env.campus_env_base import CampusEnvBase 
from env.env_wrapper import CampusEnvWrapper 
from configs import (SEED, USE_IMG, VALID_SPEED,
                     UPDATE_IMG_ENCODE, ACTOR_CONFIGS, CRITIC_CONFIGS, )


class SceneChoose(): 
    def __init__(self, trajectory_path, save_path):
        self.save_path = save_path
        with open(trajectory_path, 'r') as file:
            data = json.load(file) 
        self.case_num = len(data) 
        self.success_record = {}
        self.case_record = [] 
        self.case_success_rate = {}
        for i in range(self.case_num):
            self.case_success_rate[str(i)] = [] 
        self.horizon = 500
        
    def choose_case(self): 
        if np.random.random() < 0.2 or len(self.case_record) < self.horizon:
            return np.random.randint(0, self.case_num) 
         
        success_rate = [] 
        for i in range(self.case_num): 
            idx = str(i) 
            if len(self.case_success_rate[idx]) <= 1: 
                success_rate.append(0) 
            else:
                recent_success_record = self.case_success_rate[idx][-min(10, len(self.case_success_rate[idx])):] 
                success_rate.append(np.sum(recent_success_record)/len(recent_success_record)) 
        fail_rate = 1 - np.array(success_rate) 
        fail_rate = np.clip(fail_rate, 0.005, 1) 
        fail_rate = fail_rate / np.sum(fail_rate)  # sum to 1 
        return np.random.choice(np.arange(len(fail_rate)), p=fail_rate) 
    
    def update_success_record(self, success, case_id):
        self.case_success_rate[str(case_id)].append(success)
        self.case_record.append(case_id)
        self._log_case_success_rate() 
    
    def _log_case_success_rate(self):
        with open(os.path.join(self.save_path, 'case_success_rate.json'), 'w') as file:
            json.dump(self.case_success_rate, file, indent=4)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_ckpt', type=str, default=None)  # '/media/k/part11/workspace/rl_planner/src/log/exp/sac_20250507_214109/SAC_6.pt' 
    parser.add_argument('--img_ckpt', type=str, default=None)  # './model/ckpt/autoencoder.pt' 
    parser.add_argument('--map_path', type=str, default='../data/lanelet2_map/campus_lanelet2_map_v1.osm') 
    parser.add_argument('--trajectory_path', type=str, default='../data/trajectory/campus_trajectory_data_v1.json')
    parser.add_argument('--train_episode', type=int, default=100000)
    parser.add_argument('--eval_episode', type=int, default=2000)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=True)
    args = parser.parse_args()

    verbose = args.verbose 

    if args.visualize: 
        raw_env = CampusEnvBase(fps=100, map_path=args.map_path, trajectory_path=args.trajectory_path)
    else:
        raw_env = CampusEnvBase(fps=100, render_mode="rgb_array", map_path=args.map_path, trajectory_path=args.trajectory_path)
    env = CampusEnvWrapper(raw_env) 

    # the path to log and save model
    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = relative_path+'/log/exp/sac_%s/' % timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    scene_chooser = SceneChoose(args.trajectory_path, save_path) 

    writer = SummaryWriter(save_path)
    copyfile('./configs/configs.py', save_path+'configs.txt')
    print("You can track the training process by command 'tensorboard --logdir %s'" % save_path)

    env.action_space.seed(SEED)
    np.random.seed(SEED) 
    torch.manual_seed(SEED) 

    actor_params = ACTOR_CONFIGS 
    critic_params = CRITIC_CONFIGS 
    configs = {
        "discrete": False,
        "observation_shape": env.observation_shape,
        "action_dim": env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
    }
    print('observation_space:',env.observation_space)

    rl_agent = SAC(configs) 
    checkpoint_path = args.agent_ckpt 
    if checkpoint_path is not None:
        print(f'load pre-trained agent from {checkpoint_path}')
        rl_agent.load(checkpoint_path, params_only=True) 
    img_encoder_checkpoint = args.img_ckpt if USE_IMG else None 
    if img_encoder_checkpoint is not None and os.path.exists(img_encoder_checkpoint): 
        rl_agent.load_img_encoder(img_encoder_checkpoint, require_grad=UPDATE_IMG_ENCODE) 

    ## step_ratio = env.vehicle.kinetic_model.step_len*env.vehicle.kinetic_model.n_step*VALID_SPEED[1]
    agent = PlanningAgent(rl_agent) 

    case_id_list = []
    reward_list = []            # 한 episode의 총 보상 
    reward_per_state_list = []  # 한 step 단위의 보상 
    reward_info_list = []       # 한 episode의 각 보상 항목별 총 보상
    succ_record = []            # 성공 여부 기록 
    total_step_num = 0
    best_success_rate = 0 

    scene_info = {'mode':'normal_parking'}
    for i in range(args.train_episode):
        case_id = scene_chooser.choose_case() 
        print(f'[Epoisod: {i}] case_id: {case_id}')
        print('-'*50)

        obs = env.reset(case_id, scene_info)
        case_id_list.append(case_id) 
        done = False 
        total_reward = 0 
        step_num = 0 
        reward_info = [] 

        while not done:
            step_num += 1 
            total_step_num += 1 
            # rs 경로가 존재하지 않고 replay_memory가 충분하지 않은 경우  
            if total_step_num <= agent.configs.memory_size: 
                # explore 
                action = env.action_space.sample() 
                log_prob = agent.get_log_prob(obs, action)  # 현재 상태에 대한 행동의 확률 
            else:
                # get action from the agent
                action, log_prob = agent.get_action(obs) 
            
            # 한 step 시뮬레이션 
            next_obs, reward, done, info = env.step(action) 
            reward_info.append(list(info['reward_info'].values())) 
            agent.push_memory((obs, action, reward, done, log_prob, next_obs))  # 경험 저장 
            reward_per_state_list.append(reward) 
            total_reward += reward 
            obs = next_obs 

            if total_step_num > agent.configs.memory_size and total_step_num%10==0: 
                actor_loss, critic_loss = agent.update()
                agent.actor_loss_list.append(actor_loss) 
                agent.critic_loss_list.append(critic_loss) 

                if total_step_num%200==0:
                    print(f'actor_loss: {actor_loss}, critic_loss: {critic_loss}')
                    writer.add_scalar("actor_loss", actor_loss, i)
                    writer.add_scalar("critic_loss", critic_loss, i) 

            if done:
                print(f'Status: {info["status"]}', end=' ')
                if info['status'] == Status.ARRIVED: 
                    print('[success]')
                    succ_record.append(1) 
                    scene_chooser.update_success_record(1, case_id) 
                else:
                    print('[failed]')
                    succ_record.append(0) 
                    scene_chooser.update_success_record(0, case_id) 
            
        success_rate = np.mean(succ_record[-100:])

        writer.add_scalar("total_reward", total_reward, i) 
        writer.add_scalar("avg_reward", np.mean(reward_per_state_list[-1000:]), i) 
        writer.add_scalar("action_std0", agent.log_std.detach().cpu().numpy().reshape(-1)[0], i) 
        writer.add_scalar("action_std1", agent.log_std.detach().cpu().numpy().reshape(-1)[1], i)
        writer.add_scalar("alpha", agent.alpha.detach().cpu().numpy().reshape(-1)[0], i) 
        writer.add_scalar("success_rate", success_rate, i)  # 최근 100개 에피소드 성공률 
        writer.add_scalar("step_num", step_num, i)          # 현재 episode의 step 수 

        print(f'success rate: {np.sum(succ_record[-100:])}/{len(succ_record[-100:])} | best success rate: {best_success_rate}')
        print(f'step_num: {step_num}')
        print(f'total_reward : {total_reward} (total reward of current episode)')
        print(f'avg_reward: {np.mean(reward_per_state_list[-1000:])} (recent average reward per state)')
        print(f'action_std: {agent.log_std.detach().cpu().numpy().reshape(-1)}')
        print('')

        reward_list.append(total_reward)
        reward_info = np.sum(np.array(reward_info), axis=0) # reward 각 항목별로 합산 
        reward_info = np.round(reward_info,2)
        reward_info_list.append(list(reward_info))

        if verbose and i%10==0 and i>0:
            print('success rate: ', np.sum(succ_record),'/',len(succ_record))
            print('log_std: ', agent.log_std.detach().cpu().numpy().reshape(-1), 'alpha: ', agent.alpha.detach().cpu().numpy().reshape(-1))
            print("episode:%s  average reward:%s"%(i,np.mean(reward_list[-50:])))
            print('actor loss: ', np.mean(agent.actor_loss_list[-100:]), 'critic loss: ', np.mean(agent.critic_loss_list[-100:]))
            print('time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward')
            for j in range(10):
                print(case_id_list[-(10-j)],reward_list[-(10-j)],reward_info_list[-(10-j)])
            print("")

        # save best model 
        if success_rate > best_success_rate and i > 100: 
            agent.save("%s/SAC_best.pt" % save_path, params_only=True) 
            best_success_rate = success_rate 
            f_best_log = open(os.path.join(save_path, 'best_log.txt'), 'w')
            f_best_log.write(f'episode: {i}, step_num: {total_step_num}\n')
            f_best_log.write(f'success rate: {success_rate}\n')

        if verbose and i%20==0:
            episodes = [j for j in range(len(reward_list))]
            mean_reward = [np.mean(reward_list[max(0,j-50):j+1]) for j in range(len(reward_list))]
            plt.plot(episodes,reward_list)
            plt.plot(episodes,mean_reward)
            plt.xlabel('episodes')
            plt.ylabel('reward')
            f = plt.gcf()
            f.savefig(f'{save_path}/reward.png')
            f.clear()

        if (i+1) % 2000 == 0:
            agent.save(f"{save_path}/SAC_{i}.pt",params_only=True)

    print('done.') 


