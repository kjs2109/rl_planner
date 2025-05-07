import os 
import math 
import numpy as np 
import matplotlib.pyplot as plt 

from shapely.geometry import Polygon, MultiPolygon, LinearRing


def random_gaussian_num(mean, std, clip_low, clip_high):
    rand_num = np.random.randn()*std + mean
    return np.clip(rand_num, clip_low, clip_high)

def random_uniform_num(clip_low, clip_high):
    rand_num = np.random.rand()*(clip_high - clip_low) + clip_low
    return rand_num

def sample_straight_forward_action():
        steer = random_gaussian_num(mean=0, std=0.3, clip_low=-2*math.pi, clip_high=2*math.pi)  
        speed = random_gaussian_num(mean=1.5, std=0.3, clip_low=0.5, clip_high=2.5)   
        return np.array([steer, speed], dtype=np.float32)

def check_and_mkdir(target_path):
    print("Target_path: " + str(target_path))
    path_to_targets = os.path.split(target_path)

    path_history = '/'
    for path in path_to_targets:
        path_history = os.path.join(path_history, path)
        if not os.path.exists(path_history):
            os.mkdir(path_history) 


class DebugVisualizer:
    def __init__(self, save_path=None):
        self.save_path = save_path
    
    def figure_init(self, title="Debug Visualization", xlim=(-30, 30), ylim=(-30, 30)): 
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_aspect('equal')
        self.ax.set_xticks(np.arange(xlim[0], xlim[1]+1, 10))
        self.ax.set_yticks(np.arange(ylim[0], ylim[1]+1, 10))
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title(title)
        self.ax.grid(True)

    def _draw_text(self, x, y, text, fontsize=10):
        self.ax.text(x, y, text, fontsize=fontsize)

    def draw_arrow(self, x, y, yaw, length=2.0, color='white'):
        dx, dy = length * np.cos(yaw), length * np.sin(yaw)
        self.ax.arrow(x, y, dx, dy, head_width=0.5, head_length=0.7, fc=color, ec=color)

    def draw_linear_ring(self, linear_ring, color='blue', edgecolor='blue', alpha=1.0, label=None): 
        if isinstance(linear_ring, LinearRing): 
            ring = plt.Polygon(list(linear_ring.coords), facecolor=color, edgecolor=edgecolor, alpha=alpha, label=label)
            self.ax.add_patch(ring)
        else: 
            raise ValueError("Unsupported geometry type. Only LinearRing is supported.")

    def _draw_polygon(self, coords, closed, color, edgecolor, alpha, label):
        polygon = plt.Polygon(coords, closed=closed, facecolor=color, edgecolor=edgecolor, alpha=alpha, label=label)
        self.ax.add_patch(polygon)
    
    def draw_polygon(self, polygon, closed=True, color='gray', edgecolor="gray", alpha=1.0, label=None):
        if isinstance(polygon, Polygon): 
            self._draw_polygon(list(polygon.exterior.coords), closed, color, edgecolor, alpha, label)
        elif isinstance(polygon, MultiPolygon): 
            for geom in polygon.geoms:
                self._draw_polygon(list(geom.exterior.coords), closed, color, edgecolor, alpha, label)
        else: 
            raise ValueError("Unsupported geometry type. Only Polygon and MultiPolygon are supported.")

    def save(self, filename=None):
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            num_files = len([f for f in os.listdir(self.save_path) if os.path.isfile(os.path.join(self.save_path, f))])
            filename = filename or f'image_{num_files}.png'
            self.fig.savefig(os.path.join(self.save_path, filename))

    def show(self):
        plt.show()

    def clear(self):
        plt.clf()