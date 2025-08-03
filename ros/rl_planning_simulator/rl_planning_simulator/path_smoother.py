import cvxpy as cp
import numpy as np
import math

def smooth_trajectory(trajectory, smooth_weight=1000.0, lat_error_weight=500.0):
    if len(trajectory) < 3:
        return trajectory 

    xy = np.array([[pt[0], pt[1]] for pt in trajectory])
    N = len(xy)

    x = cp.Variable(N)
    y = cp.Variable(N)
    obj = []

    # Lateral error term
    for i in range(N):
        obj.append(lat_error_weight * cp.square(x[i] - xy[i, 0]))
        obj.append(lat_error_weight * cp.square(y[i] - xy[i, 1]))

    # Smoothness term (second derivative)
    for i in range(1, N - 1):
        obj.append(smooth_weight * cp.square(x[i - 1] - 2 * x[i] + x[i + 1]))
        obj.append(smooth_weight * cp.square(y[i - 1] - 2 * y[i] + y[i + 1]))

    # Constraints: start and end fixed
    constraints = [
        x[0] == xy[0, 0], y[0] == xy[0, 1],
        x[N - 1] == xy[-1, 0], y[N - 1] == xy[-1, 1]
    ]

    cp.Problem(cp.Minimize(cp.sum(obj)), constraints).solve(solver=cp.OSQP)

    smoothed_xy = np.vstack([x.value, y.value]).T

    new_traj = []
    for i in range(N):
        x_i, y_i = smoothed_xy[i]
        if i < N - 1:
            dx = smoothed_xy[i + 1, 0] - x_i
            dy = smoothed_xy[i + 1, 1] - y_i
            yaw = math.atan2(dy, dx)
        else:  
            yaw = new_traj[-1][3] if new_traj else trajectory[-1][3]

        v = trajectory[i][2]  
        new_traj.append((x_i, y_i, v, yaw))

    return new_traj
