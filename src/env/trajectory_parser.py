import math
import json 
from shapely.geometry import Point 
from shapely.affinity import rotate, translate 

class TrajectoryParser:
    def __init__(self, trajectory_path):
        self.trajectory_path = trajectory_path
        self.trajectory_data = []
        self._load_trajectory()

    def _load_trajectory(self):
        with open(self.trajectory_path, 'r') as file: 
            data = json.load(file) 
            for sample in data:
                trajectory = sample['synced_sample']['trajectory']
                odometry = sample['synced_sample']['odometry']
                self.trajectory_data.append((trajectory, odometry))

    def _get_trajectory_by_id(self, id):
        if id < 0 and id >= len(self.trajectory_data):
            raise IndexError("ID out of range") 
        
        sample = self.trajectory_data[id] 
        trajectory = sample[0] 
        odometry = sample[1] 

        return trajectory, odometry 
    
    def _traj_map_to_ego(self, center, trajectory):
        cx, cy, heading = center
        trajectory_points = []

        for traj in trajectory[::10]:
            point = traj['trajectory_point']
            x = point['position']['x']
            y = point['position']['y']
            yaw_p = math.radians(point['yaw_deg'])
            yaw_h = math.pi/2 - heading 
            yaw_traj_p = (yaw_p - yaw_h) + math.pi/2

            pt = Point(x, y)
            translated_pt = translate(pt, xoff=-cx, yoff=-cy) 
            rotated_pt = rotate(translated_pt, heading, origin=(0, 0), use_radians=True)

            trajectory_points.append((rotated_pt.x, rotated_pt.y, yaw_traj_p))

        return trajectory_points
    
    def get_processed_trajectory(self, id): 
        selected_trajectory, _ = self._get_trajectory_by_id(id)
        start_point = selected_trajectory[0]['trajectory_point']
        x = start_point['position']['x']
        y = start_point['position']['y'] 
        yaw = -math.radians(start_point['yaw_deg'])+math.pi/2
        processed_trajectory = self._traj_map_to_ego((x, y, yaw), selected_trajectory) 

        return processed_trajectory
    
    def get_ori_center_point(self, id):
        selected_trajectory, _ = self._get_trajectory_by_id(id)
        start_point = selected_trajectory[0]['trajectory_point']
        x = start_point['position']['x']
        y = start_point['position']['y'] 
        yaw = math.radians(start_point['yaw_deg'])

        return (x, y, yaw)

    