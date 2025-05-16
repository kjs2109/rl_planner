from typing import DefaultDict 
import pickle 

import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import trange 

from env.vehicle import Status 
from configs import (TOLERANT_TIME) 


def eval_agent(env, agent, episode=2000, log_path='', multi_mode=False, mode='normal_parking'): 

    succ_record = [] 
    reward_record = []
    step_record = DefaultDict(list)
    path_length_record = DefaultDict(list) 
    success_step_record = [] 
    eval_record = []
    if multi_mode: 
        succ_rate_mode = DefaultDict(list) 
        step_num_mode = DefaultDict(list) 
        path_length_mode = DefaultDict(list) 
    succ_rate_case = DefaultDict(list)
    reward_case = DefaultDict(list) 

    scene_info = {'mode': mode, 'flip_prob': 0.0}
    for case_id in trange(episode): 
        obs = env.reset(case_id, scene_info) 
        done = False 
        total_reward = 0 
        step_num = 0  
        path_length = 0 
        last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)
        last_target_obs = obs['target']

        while not done: 
            step_num += 1 
            
            # action 
            action, _ = agent.get_action(obs)  # without action mask 
            if (last_target_obs == obs['target']).all(): # 이전 target 관측치와 현재 target 관측치가 같다면 
                action = env.action_space.sample()  # random action 

            # get next observation 
            next_obs, reward, done, info = env.step(action)

            # update state 
            last_target_obs = obs['target'] 
            obs = next_obs
            total_reward += reward 
            path_length += np.linalg.norm(np.array(last_xy)-np.array((env.vehicle.state.loc.x, env.vehicle.state.loc.y)))
            last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)

            if done: 
                if info['status'] == Status.ARRIVED: 
                    succ_record.append(1) 
                else: 
                    succ_record.append(0)

        # record results 
        reward_record.append(total_reward)
        succ_rate_case[case_id].append(succ_record[-1]) 
        reward_case[case_id].append(reward_record[-1])
        if step_num < TOLERANT_TIME: 
            path_length_record[case_id].append(path_length) 
        if info['status'] == Status.OUTBOUND: 
            step_record[case_id].append(TOLERANT_TIME)
        else: 
            step_record[case_id].append(step_num)
        if succ_record[-1] == 1: 
            success_step_record.append(step_num) 

        eval_record.append({'case_id':case_id,
                            'status':info['status'],
                            'step_num':step_num,
                            'reward':total_reward,
                            'path_length':path_length,
                            })

    print('#'*15) 
    print('EVALUDATION RESULT') 
    print('success rate: ', np.mean(succ_record))
    print('average reward: ', np.mean(reward_record)) 
    print('-'*10) 
    print('success rate per case: ') 
    case_ids = [int(k) for k in succ_rate_case.keys()] 
    case_ids.sort() 
    if len(case_ids) < 10: 
        print('-'*10) 
        print('average reward per case: ') 
        for k in case_ids: 
            print(f'case {k} scuccess rate: {np.mean(succ_rate_case[k])}')
            print(f'case {k} average reward: {np.mean(reward_case[k])} +-({np.std(reward_case[k])})')

    if multi_mode: 
        print('success rate per mode: ') 
        for k in succ_rate_mode.keys(): 
            print(f'{k} (case num {len(succ_rate_mode[k])}): ({np.mean(succ_rate_mode[k])})') 

    if log_path is not None: 
        def plot_time_ratio(node_list): 
            max_node = TOLERANT_TIME 
            raw_len = len(node_list) 
            filtered_node_list = [] 
            for n in node_list: 
                if n != max_node: 
                    filtered_node_list.append(n) 
            filtered_node_list.sort() 
            ratio_list = [i/raw_len for i in range(1, len(filtered_node_list)+1)] 
            plt.plot(filtered_node_list, ratio_list) 
            plt.xlabel('scearch node') 
            plt.ylabel('accumulate success rate') 
            fig = plt.gcf() 
            fig.savefig(log_path+'/success_rate.png') 
            plt.close() 

        all_step_record = [] 
        for k in step_record.keys(): 
            all_step_record += step_record[k] 
        plot_time_ratio(all_step_record) 

        # save eval result 
        f_record_txt = open(log_path+'/result.txt', 'w', newline='')
        f_record_txt.write('success rate: %s\n'%np.mean(succ_record))
        f_record_txt.write('step num: %s '%np.mean(success_step_record)+'+-(%s)\n'%np.std(success_step_record))
        if multi_mode:
            f_record_txt.write('\n')
            for k in succ_rate_mode.keys():
                f_record_txt.write(f'{k} (case num {len(succ_rate_mode[k])}): {np.mean(succ_rate_mode[k])}\n')
                f_record_txt.write(f'step_num: {np.mean(step_record[k])} +-({np.std(step_record[k])})\n')
                f_record_txt.write(f'path length: {np.mean(path_length_record[k])} +-({np.std(path_length_record[k])})\n')

        if len(case_ids) < 10: 
            f_record_txt.write('\n')
            for k in case_ids:
                f_record_txt.write(f'case {k} success rate: {np.mean(succ_rate_case[k])}\n')
                f_record_txt.write(f'case {k} average reward: {np.mean(reward_case[k])} +-({np.std(reward_case[k])})\n')
                f_record_txt.write(f'case {k} step num: {np.mean(step_record[k])} +-({np.std(step_record[k])})\n')
                f_record_txt.write(f'case {k} path length: {np.mean(path_length_record[k])} +-({np.std(path_length_record[k])})\n')
        
        f_record_txt.close() 

    return np.mean(succ_record)

