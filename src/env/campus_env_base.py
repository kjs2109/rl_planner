from typing import Optional
import math 
from typing import OrderedDict 

import numpy as np 
import gym 
from gym import spaces 
from gym.error import DependencyNotInstalled 
from shapely.geometry import Polygon 
from shapely.affinity import affine_transform 
try:
    import pygame 
except ImportError:
    raise DependencyNotInstalled("pygame is not installed, run `pip install pygame` to install it.") 

from env.vehicle import Vehicle, Status, State  
from env.map_base import Area 
from env.lidar_simulator import LidarSimlator  
from env.campus_map import CampusMap 
from env.observation_processor import Obs_Processor 
# import env.reeds_shepp as rsCurve 
# from model.action_mask import ActionMask 

from configs import (
    WIN_W, WIN_H, FPS,
    USE_IMG, USE_LIDAR,
    WIN_W, WIN_H, K, BG_COLOR,
    OBS_W, OBS_H,
    LIDAR_RANGE, LIDAR_NUM,
    START_COLOR, DEST_COLOR,
    TRAJ_RENDER_LEN, TRAJ_COLORS,
    VALID_SPEED, VALID_STEER,
    NUM_STEP, STEP_LENGTH,
    MAX_DIST_TO_DEST, TOLERANT_TIME, 
    OBSTACLE_COLOR, NON_DRIVABLE_COLOR,
    RENDER_TRAJ, 
)

class CampusEnvBase(gym.Env):
    metadata = {"render_mode": ["human", "rgb_array"]} 

    def __init__(
            self, 
            render_mode:str = None, 
            fps:int = FPS, 
            verbose:bool = True, 
            use_lidar_observation:bool = USE_LIDAR, 
            use_img_observation:bool = USE_IMG,
            # use_action_mask:bool = USE_ACTION_MASK, 
            map_path:str = None,
            trajectory_path:str = None): 
        super().__init__() 

        self.verbose = verbose
        self.use_lidar_observation = use_lidar_observation
        self.use_img_observation = use_img_observation
        # self.use_action_mask = use_action_mask
        self.render_mode = "human" if render_mode is None else render_mode
        self.fps = fps
        self.screen:Optional[pygame.Surface] = None 
        self.matrix = None 
        self.clock = None 
        self.is_open = True 
        self.t = 0.0 
        self.k = None 
        # self.level = MAP_LEVEL 
        self.tgt_repr_size = 5

        self.map = CampusMap(map_path, trajectory_path) 
        self.vehicle = Vehicle(n_step=NUM_STEP, step_len=STEP_LENGTH)
        self.lidar = LidarSimlator(LIDAR_RANGE, LIDAR_NUM)
        self.reward = 0.0
        self.prev_reward = 0.0
        self.accum_arrive_reward = 0.0

        # action space 
        self.action_space = spaces.Box(
            np.array([VALID_STEER[0], VALID_SPEED[0]]).astype(np.float32),  # low 
            np.array([VALID_STEER[1], VALID_SPEED[1]]).astype(np.float32),  # high 
        ) # steer, speed

        # observation(state) space 
        self.observation_space = {} 
        if self.use_img_observation:
            self.img_processor = Obs_Processor() 
            self.observation_space['img'] = spaces.Box(
                low=0, 
                high=255, 
                shape=(OBS_W//self.img_processor.downsample_rate,  # default=4
                       OBS_H//self.img_processor.downsample_rate,
                       self.img_processor.n_channels), dtype=np.uint8
            )
            self.raw_img_shape = (OBS_W, OBS_H, 3)

        if self.use_lidar_observation: 
            low_bound, high_bound = np.zeros((LIDAR_NUM)), np.ones((LIDAR_NUM)) * LIDAR_RANGE 
            self.observation_space['lidar'] = spaces.Box(
                low=low_bound, high=high_bound, shape=(LIDAR_NUM, ), dtype=np.float32
            )

        # d, cos(θ), sin(θ), cos(Δ), sin(Δ)
        low_bound = np.array([0,-1,-1,-1,-1])
        high_bound = np.array([MAX_DIST_TO_DEST,1,1,1,1])
        self.observation_space['target'] = spaces.Box(
            low=low_bound, high=high_bound, shape=(self.tgt_repr_size,), dtype=np.float64
        )

    def reset(self, case_id:int=None, scene_info:dict=None) -> np.ndarray: 
        self.reward = 0.0 
        self.prev_reward = 0.0 
        self.accum_arrive_reward = 0.0 
        self.t = 0.0 
        if case_id is not None: 
            initial_state = self.map.reset(case_id=case_id, scene_info=scene_info) 
        else: 
            initial_state = self.map.reset(scene_info=scene_info)
        self.vehicle.reset(initial_state) 
        self.matrix = self.coord_transform_matrix() 
        return self.step()[0]

    # def coord_transform_matrix(self) -> list:
    #     """Get the transform matrix that convert the real world coordinate to the pygame coordinate.
    #      [k 0 bx
    #       0 k by]
    #     """
    #     k = K
    #     bx = 0.5 * (WIN_W - k * (self.map.xmax + self.map.xmin))
    #     by = 0.5 * (WIN_H - k * (self.map.ymax + self.map.ymin))
    #     self.k = k
    #     return [k, 0, 0, k, bx, by]

    # def coord_transform_matrix(self) -> list:
    #     k = K
    #     bx = 0.5 * (WIN_W - k * (self.map.xmax + self.map.xmin))
    #     by = 0.5 * (WIN_H + k * (self.map.ymax + self.map.ymin))  # +k
    #     self.k = k
    #     return [k, 0, 0, -k, bx, by] 

    def coord_transform_matrix(self) -> list:
        k = K
        ego_x = self.vehicle.state.loc.x
        ego_y = self.vehicle.state.loc.y

        # 차량 위치를 화면 중앙에 놓기 위한 보정
        bx = WIN_W / 2 - k * ego_x
        by = (WIN_H * 2 / 3) + k * ego_y  # y축 반전 고려

        self.k = k
        return [k, 0, 0, -k, bx, by] 

    def _coord_transform(self, object) -> list: 
        if hasattr(object, "shape"): 
            transformed = affine_transform(object.shape, self.matrix)
            return list(transformed.exterior.coords) 
        else: 
            transformed = affine_transform(object, self.matrix) 
            return list(transformed.coords)
    
    def _detect_collision(self):
        for obstacle in self.map.obstacles: 
            if self.vehicle.box.intersects(obstacle.shape):
                return True 
        for non_drivable_area in self.map.zoomed_non_drivable_area:
            if self.vehicle.box.intersects(non_drivable_area.shape):
                return True
        return False 

    def _detect_outbounds(self):
        x, y = self.vehicle.state.loc.x, self.vehicle.state.loc.y 
        return x>self.map.xmax or x<self.map.xmin or y>self.map.ymax or y<self.map.ymin

    def _check_arrived(self):
        vehicle_box = Polygon(self.vehicle.box) 
        dest_box = Polygon(self.map.dest_box) 
        union_area = vehicle_box.intersection(dest_box).area
        if union_area / dest_box.area > 0.95:
            return True
        return False

    def _check_time_exceeded(self): 
        return self.t > TOLERANT_TIME  
    
    def _check_status(self):
        if self._detect_collision(): 
            return Status.COLLIDED 
        if self._detect_outbounds(): 
            return Status.OUTBOUND 
        if self._check_arrived(): 
            return Status.ARRIVED 
        if self._check_time_exceeded(): 
            return Status.OUTTIME 
        return Status.CONTINUE 
    
    def _get_reward(self, prev_state:State, curr_state:State):
        # time penalty 
        time_cost = -np.tanh(self.t / (10*TOLERANT_TIME)) 

        # RS distance reward 
        rs_dist_reward = 0.0 

        def get_angle_diff(angle1, angle2):
            angle_dif = math.acos(math.cos(angle1 - angle2))  # absolute angle difference 
            return angle_dif if angle_dif < math.pi/2 else math.pi - angle_dif  

        # curr 
        dist_diff = curr_state.loc.distance(self.map.dest.loc)
        angle_diff = get_angle_diff(curr_state.heading, self.map.dest.heading) 
        # prev 
        prev_dist_diff = prev_state.loc.distance(self.map.dest.loc)
        prev_angle_diff = get_angle_diff(prev_state.heading, self.map.dest.heading)

        # reward 
        dist_norm_ratio = max(self.map.dest.loc.distance(self.map.start.loc), 10) 
        angle_norm_ratio = math.pi 
        dist_reward = prev_dist_diff/dist_norm_ratio - dist_diff/dist_norm_ratio 
        angle_reward = prev_angle_diff/angle_norm_ratio - angle_diff/angle_norm_ratio 

        # Box union reward 
        vehicle_box = Polygon(self.vehicle.box) 
        dest_box = Polygon(self.map.dest_box) 
        union_area = vehicle_box.intersection(dest_box).area 
        box_union_reward = union_area / (2*dest_box.area - union_area)  # IoU 
        if box_union_reward < self.accum_arrive_reward: 
            box_union_reward = 0 
        else: 
            prev_arrive_reward = self.accum_arrive_reward 
            self.accum_arrive_reward = box_union_reward  
            box_union_reward -= prev_arrive_reward  # 증가량 만큼 보상  
        return [time_cost, rs_dist_reward, dist_reward, angle_reward, box_union_reward] 

    def get_reward(self, status, prev_status):
        reward_info = [0, 0, 0, 0, 0] 
        if status == Status.CONTINUE: 
            reward_info = self._get_reward(prev_status, self.vehicle.state) 
        return reward_info 

    def step(self, action:np.array = None):
        '''
        Parameters:
        ----------
        `action`: `np.ndarray`

        Returns:
        ----------
        ``observation`` (Dict): the observation of image based surroundings, lidar view and target representation.
                                If `use_lidar_observation` is `True`, then `obsercation['img'] = None`.
                                If `use_lidar_observation` is `False`, then `obsercation['lidar'] = None`. 
        ``reward_info`` (OrderedDict): different types of reward information, including: time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward
        ``status`` (`Status`): represent the state of vehicle, including: `CONTINUE`, `ARRIVED`, `COLLIDED`, `OUTBOUND`, `OUTTIME`
        ``info`` (`OrderedDict`): other information.
        '''
        assert self.vehicle is not None, "The vehicle is not initialized." 
        prev_state = self.vehicle.state 
        collide = False 
        arrive = False 
        if action is not None:
            for simu_step_num in range(NUM_STEP): 
                # 현재 action으로 NUM_STEP 만큼 vehicle 이동 
                prev_info = self.vehicle.step(action, step_time=1) 
                if self._check_arrived(): 
                    arrive = True 
                    break 
                if self._detect_collision(): 
                    if simu_step_num == 0:
                        collide = False  # ENV_COLLIDE 
                        self.vehicle.retreat(prev_info) 
                    else: 
                        self.vehicle.retreat(prev_info)     
                    simu_step_num -= 1 
                    break

            simu_step_num += 1 
            if simu_step_num: 
                del self.vehicle.trajectory[-simu_step_num:-1]  # remove redundant trajectory

        self.t += 1  # time step 
        observation = self.render(self.render_mode)  
        if arrive:
            status = Status.ARRIVED 
        else: 
            status = Status.COLLIDED if collide else self._check_status() 

        reward_list = self.get_reward(status, prev_state)  
        reward_info = OrderedDict({
                'time_cost': reward_list[0],
                'rs_dist_reward': reward_list[1],
                'dist_reward': reward_list[2],
                'angle_reward': reward_list[3],
                'box_union_reward': reward_list[4]
            })
        info = OrderedDict({'reward_info': reward_info, 'status': status})

        # if self.t > 1 and status==Status.CONTINUE: # and self.vehicle.state.loc.distance(self.map.dest.loc) < RS_MAX_DIST: 
        #     rs_path_to_dest = self.rind_rs_path(status) 
        #     if rs_path_to_dest is not None: 
        #         info['path_to_dest'] = rs_path_to_dest 

        return observation, reward_info, status, info 

    def _render(self, surface:pygame.Surface): 
        surface.fill(BG_COLOR) 
        
        for non_drivable_area in self.map.zoomed_non_drivable_area:
            pygame.draw.polygon(surface, NON_DRIVABLE_COLOR, self._coord_transform(non_drivable_area))
        # for drivable_area in self.map.zoomed_drivable_area:
        #     pygame.draw.polygon(surface, drivable_area.color, self._coord_transform(drivable_area), width=1)
        for obstacle in self.map.obstacles: 
            pygame.draw.polygon(surface, OBSTACLE_COLOR, self._coord_transform(obstacle))

        pygame.draw.polygon(surface, START_COLOR, self._coord_transform(self.map.start_box), width=1) 
        pygame.draw.polygon(surface, DEST_COLOR, self._coord_transform(self.map.dest_box), width=1)
        pygame.draw.polygon(surface, self.vehicle.color, self._coord_transform(self.vehicle.box))

        if RENDER_TRAJ and len(self.vehicle.trajectory) > 1:
            render_len = min(len(self.vehicle.trajectory), TRAJ_RENDER_LEN)
            for i in range(render_len):
                vehicle_box = self.vehicle.trajectory[-(render_len-i)].create_box()  # trajectory.creat_box ?? 
                # pygame.draw.polygon(surface, TRAJ_COLORS[-(render_len-i)], self._coord_transform(vehicle_box)) 
                points = self._coord_transform(vehicle_box)
                temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
                pygame.draw.polygon(temp_surface, TRAJ_COLORS[-(render_len-i)], points)
                surface.blit(temp_surface, (0, 0))
                # # 2. Compute rear axle center from box points
                # rear_x = (points[0][0] + points[3][0]) / 2
                # rear_y = (points[0][1] + points[3][1]) / 2
                # # 2. draw trajectory center point as a small circle
                # center = (rear_x, rear_y) # self._coord_transform_point(rear_x, rear_y)
                # pygame.draw.circle(surface, (150, 200, 10), center, 2)  # red dot, radius=2

    def _get_img_observation(self, surface:pygame.Surface): 
        angle = self.vehicle.state.heading 
        old_center = surface.get_rect().center 

        capture = pygame.transform.rotate(surface, np.rad2deg(angle)) 
        rotate = pygame.Surface((WIN_W, WIN_H)) 
        rotate.blit(capture, capture.get_rect(center=old_center)) 

        vehicle_center = np.array(self._coord_transform(self.vehicle.box.centroid)[0]) 
        dx = (vehicle_center[0]-old_center[0])*np.cos(angle) + (vehicle_center[1]-old_center[1])*np.sin(angle) 
        dy = -(vehicle_center[0]-old_center[0])*np.sin(angle) + (vehicle_center[1]-old_center[1])*np.cos(angle) 

        observation = pygame.Surface((WIN_W, WIN_H)) 
        observation.blit(rotate, (int(-dx), int(-dy))) 
        observation = observation.subsurface(((WIN_W-OBS_W)/2, (WIN_H-OBS_H)/2), (OBS_W, OBS_H))

        obs_str = pygame.image.tostring(observation, "RGB")
        observation = np.frombuffer(obs_str, dtype=np.uint8)
        observation = observation.reshape(self.raw_img_shape)

        return observation

    def _process_img_observation(self, img):
        processed_img = self.img_processor.process_img(img)
        return processed_img

    def _get_lidar_observation(self): 
        obs_list = [obs.shape for obs in self.map.obstacles]
        lidar_view = self.lidar.get_observation(self.vehicle.state, obs_list)
        return lidar_view

    def _get_targt_repr(self): 
        # target position representation
        dest_pos = (self.map.dest.loc.x, self.map.dest.loc.y, self.map.dest.heading)
        ego_pos = (self.vehicle.state.loc.x, self.vehicle.state.loc.y, self.vehicle.state.heading)
        rel_distance = math.sqrt((dest_pos[0]-ego_pos[0])**2 + (dest_pos[1]-ego_pos[1])**2)
        rel_angle = math.atan2(dest_pos[1]-ego_pos[1], dest_pos[0]-ego_pos[0]) - ego_pos[2]
        rel_dest_heading = dest_pos[2] - ego_pos[2]
        tgt_repr = np.array([rel_distance, math.cos(rel_angle), math.sin(rel_angle), math.cos(rel_dest_heading), math.cos(rel_dest_heading)])
        return tgt_repr 

    def render(self, mode:str = "human"): 
        assert mode in self.metadata["render_mode"] 
        assert self.vehicle is not None 

        # pygame 초기 설정 
        if mode == "human":
            display_flags = pygame.SHOWN 
        else: 
            display_flags = pygame.HIDDEN 
        if self.screen is None:
            pygame.init() 
            pygame.display.init() 
            self.screen = pygame.display.set_mode((WIN_W, WIN_H), flags = display_flags)
        if self.clock is None: 
            self.clock = pygame.time.Clock() 

        self._render(self.screen)  

        observation = {'img': None, 'lidar': None, 'target': None} 
        if self.use_img_observation:
            raw_observation = self._get_img_observation(self.screen) 
            observation['img'] = self._process_img_observation(raw_observation) 
        if self.use_lidar_observation: 
            observation['lidar'] = self._get_lidar_observation() 

        observation['target'] = self._get_targt_repr() 
        pygame.display.update() 
        self.clock.tick(self.fps) 

        return observation 

    def is_traj_valid(self, traj):
        pass 

    def close(self):
        if self.screen is not None:
            pygame.display.quit() 
            self.is_open = False 
            pygame.quit() 

if __name__ == "__main__":
    import time
    from utils import sample_straight_forward_action
    

    map_path = '/home/k/rl_planner_mod/data/lanelet2_map/campus_lanelet2_map.osm' 
    trajectory_path = "/home/k/rl_planner_mod/data/trajectory/synced_trajectory_odometry_v1.json"
    env = CampusEnvBase(render_mode="human", map_path=map_path, trajectory_path=trajectory_path)

    case_id = 0
    for case_id in range(0, 300, 10):
        obs = env.reset(case_id)
        done = False
        step_count = 0

        while True:
            # 랜덤한 액션 생성: [steer, speed]
            # action = env.action_space.sample()
            action = sample_straight_forward_action()

            # 환경에 한 스텝 적용
            obs, reward_info, status, info = env.step(action)

            # 출력
            # obs, reward_info, status, info = env.step()
            print("LIDAR:", obs['lidar'].shape)       # if use_lidar_observation is True
            print("Image:", obs['img'].shape)   # if use_img_observation is True
            print("Action:", action)
            print("Target repr:", obs['target'])
            print(f"Step: {step_count:3d}, Status: {status.name}, Reward: {reward_info}")
            print('--'*50)

            # 종료 조건
            if status != Status.CONTINUE:
                print(f"Episode finished with status: {status.name}")
                break

            step_count += 1
            time.sleep(0.05)

    env.close()