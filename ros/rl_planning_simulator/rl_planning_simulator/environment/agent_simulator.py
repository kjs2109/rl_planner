import math 
import pygame
import numpy as np
from shapely.geometry import Polygon, Point 
from shapely.affinity import affine_transform, translate, rotate 

from environment.vehicle import State, Status, Vehicle
from environment.map_parser import LaneletMapParser
from environment.lidar_simulator import LidarSimlator
from environment.observation_processor import Obs_Processor
from model.agent.sac_agent import SACAgent as SAC
from configs import (LIDAR_RANGE, LIDAR_NUM, WIN_W, WIN_H, OBS_W, OBS_H, NUM_STEP,
                     ACTOR_CONFIGS, CRITIC_CONFIGS, K)


class AgentSimulator():
    def __init__(self, map_path, agent_path): 

        self.map_parser = LaneletMapParser(map_path)
        self.rl_agent = self._load_agent(agent_path) 
        self.img_processor = Obs_Processor() 
        self.lidar = LidarSimlator(LIDAR_RANGE, LIDAR_NUM)

        self.matrix = self.coord_transform_matrix()

    def coord_transform_matrix(self) -> list:
        k = K
        ego_x = 0 
        ego_y = 0
        bx = WIN_W / 2 - k * ego_x
        by = (WIN_H * 2 / 3) + k * ego_y  
        self.k = k
        return [k, 0, 0, -k, bx, by] 

    def _coord_transform(self, object) -> list: 
        if hasattr(object, "shape"): 
            transformed = affine_transform(object.shape, self.matrix)
            return list(transformed.exterior.coords) 
        else: 
            transformed = affine_transform(object, self.matrix) 
            return list(transformed.coords)

    def _load_agent(self, agent_path): 
        checkpoint_path = agent_path
        actor_params = ACTOR_CONFIGS
        critic_params = CRITIC_CONFIGS
        configs = {
            "discrete": False,
            "observation_shape": {'img': (3, 128, 128), 'lidar': (200,), 'target': (5,)},
            "action_dim": 2,
            "hidden_size": 64,
            "activation": "tanh",
            "dist_type": "gaussian",
            "save_params": False,
            "actor_layers": actor_params,
            "critic_layers": critic_params,
        }
        agent = SAC(configs) 
        agent.load(checkpoint_path, params_only=True)
        print('Agent loaded successfully')
        return agent 
    
    def get_scene(self, curr_pose, zoom_width=60, zoom_height=80): 
        return self.map_parser.get_semantic_areas(curr_pose, zoom_width, zoom_height)

    def reset(self, screen, processed_area, processed_obs, curr_pose, goal_center): 
        self.t = 0 
        self.initial_pose = curr_pose
        self.vehicle = Vehicle() 
        self.vehicle.reset(State([0, 0, math.pi/2]))
        self.screen = screen
        self.non_drivable_area = processed_area  # Area
        self.processed_obs = processed_obs       # Area 
        self.goal_center = goal_center 

    def _ego_coord_transform_map(self, base_pose, target_pose):
        base_x, base_y, base_yaw = base_pose
        target_x, target_y, target_yaw = target_pose
        rotated_pt = rotate(Point(target_x, target_y), base_yaw - math.pi/2, origin=(0, 0), use_radians=True)
        map_pt = translate(rotated_pt, xoff=base_x, yoff=base_y)
        yaw = target_yaw - math.pi/2 + base_yaw
        return (map_pt.x, map_pt.y, yaw)

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
        observation = observation.reshape(OBS_W, OBS_H, 3)

        return observation

    def _process_img_observation(self, img):
        processed_img = self.img_processor.process_img(img).transpose(2, 0, 1) 
        return processed_img
    
    def _get_lidar_observation(self): 
        obs_list = [obs.shape for obs in self.processed_obs]
        obs_list += [obs.shape for obs in self.non_drivable_area]
        lidar_view = self.lidar.get_observation(self.vehicle.state, obs_list)
        return lidar_view

    def _get_targt_repr(self): 
        # target position representation
        dest_pos = self.goal_center 
        ego_pos = (self.vehicle.state.loc.x, self.vehicle.state.loc.y, self.vehicle.state.heading)
        rel_distance = math.sqrt((dest_pos[0]-ego_pos[0])**2 + (dest_pos[1]-ego_pos[1])**2)
        rel_angle = math.atan2(dest_pos[1]-ego_pos[1], dest_pos[0]-ego_pos[0]) - ego_pos[2]
        rel_dest_heading = dest_pos[2] - ego_pos[2]
        tgt_repr = np.array([rel_distance, math.cos(rel_angle), math.sin(rel_angle), math.cos(rel_dest_heading), math.cos(rel_dest_heading)])
        return tgt_repr 

    def _get_observation(self): 
        observation = {'img': None, 'lidar': None, 'target': None} 
        raw_observation = self._get_img_observation(self.screen) 
        observation['img'] = self._process_img_observation(raw_observation) 
        observation['lidar'] = self._get_lidar_observation() 
        observation['target'] = self._get_targt_repr() 
        return observation 
    
    def _detect_collision(self):
        for obstacle in self.processed_obs: 
            if self.vehicle.box.intersects(obstacle.shape):
                return True 
        for non_drivable_area in self.non_drivable_area:
            if self.vehicle.box.intersects(non_drivable_area.shape):
                return True
        return False 

    def _check_arrived(self):
        vehicle_box = Polygon(self.vehicle.box) 
        dest_box = Polygon(State(self.goal_center).create_box()) 
        union_area = vehicle_box.intersection(dest_box).area
        if union_area / dest_box.area > 0.8:
            return True
        return False
    
    def _check_time_exceeded(self): 
        return self.t > 200 # TOLERANT_TIME  
    
    def _check_status(self):
        if self._detect_collision(): 
            return Status.COLLIDED 
        if self._check_arrived(): 
            return Status.ARRIVED 
        if self._check_time_exceeded(): 
            return Status.OUTTIME 
        return Status.CONTINUE 
    
    def run(self):
        done = False  
        rl_trajectory = []
        while not done:
            observation = self._get_observation()
            action, _ = self.rl_agent.get_action(observation) 
            # if (self.processed_obs == observation['target']).all():
            #     action = self.env.action_space.sample()
     
            collide = False 
            arrive = False 
            if action is not None:
                for simu_step_num in range(NUM_STEP):  
                    prev_info = self.vehicle.step(action, step_time=1) 

                    if self._check_arrived(): 
                        arrive = True 
                        break 
                    if self._detect_collision():  
                        if simu_step_num == 0:
                            collide = False
                            self.vehicle.retreat(prev_info) 
                        else: 
                            self.vehicle.retreat(prev_info)     
                        simu_step_num -= 1 
                        break

                simu_step_num += 1 
                if simu_step_num: 
                    del self.vehicle.trajectory[-simu_step_num:-1] 
            
            self.t += 1
            if arrive:
                status = Status.ARRIVED 
                for traj_state in self.vehicle.trajectory: 
                    x, y = traj_state.loc.x, traj_state.loc.y 
                    heading = traj_state.heading 
                    rl_trajectory.append(self._ego_coord_transform_map(self.initial_pose, (x, y, heading)))
                done = True 
            else:
                status = Status.COLLIDED if collide else self._check_status()  

            if status == Status.OUTTIME: 
                done = True 

        return rl_trajectory