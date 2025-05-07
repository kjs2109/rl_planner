import numpy as np 

DEFAULT_UPDATE_MODAL = {'img': False, 'lidar': True, 'target': True, 'action_mask': False} 

class StateNorm():
    def __init__(self, observation_shape:dict, update_modal:dict=DEFAULT_UPDATE_MODAL) -> None:
        self.observation_shape = observation_shape 
        self.update_modal = update_modal 
        self.n_state = 0 
        self.state_mean, self.S, self.state_std = {}, {}, {} 
        for obs_type in observation_shape.keys():
            self.state_mean[obs_type] = np.zeros(self.observation_shape[obs_type], dtype=np.float32)  # 현재 observation(관측값)의 평균 
            self.S[obs_type] = np.zeros(self.observation_shape[obs_type], dtype=np.float32)           # 현재 observation(관측값)의 분산 
            self.state_std[obs_type] = np.zeros(self.observation_shape[obs_type], dtype=np.float32)   # 현재 observation(관측값)의 표준편차
        self.fixed = False 

    def fix_parameters(self):
        self.fixed = True 

    def init_state_norm(self, mean, std, S, n_state):
        self.n_state = n_state 
        self.mean, self.std, self.S = mean, std, S 

    def state_norm(self, observation:dict, update=False):
        # 초기화 
        if self.n_state == 0:
            self.n_state += 1 
            for obs_type in self.observation_shape.keys(): 
                if self.update_modal[obs_type]:   
                    self.state_mean[obs_type] = observation[obs_type] 
                    self.state_std[obs_type] = observation[obs_type] 
                    observation[obs_type] = (observation[obs_type] - self.state_mean[obs_type]) / (self.state_std[obs_type] + 1e-8)
        # 평균, 분산, 표준편차 업데이트 x 
        elif update==False or self.fixed: 
            for obs_type in self.observation_shape.keys():
                if self.update_modal[obs_type]:
                    observation[obs_type] = (observation[obs_type] - self.state_mean[obs_type]) / (self.state_std[obs_type] + 1e-8) 
        # online으로 평균, 분산, 표준편차 업데이트 후 observation 정규화 
        elif update==True: 
            self.n_state += 1 
            for obs_type in self.observation_shape.keys(): 
                if self.update_modal[obs_type]: 
                    old_mean = self.state_mean[obs_type].copy()
                    self.state_mean[obs_type] = old_mean + (observation[obs_type] - old_mean) / self.n_state 
                    self.S[obs_type] = self.S[obs_type] + (observation[obs_type] - old_mean) * (observation[obs_type] - self.state_mean[obs_type]) 
                    self.state_std[obs_type] = np.sqrt(self.S[obs_type] / self.n_state) 
                    observation[obs_type] = (observation[obs_type] - self.state_mean[obs_type]) / (self.state_std[obs_type] + 1e-8)

        return observation 