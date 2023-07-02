import torch
import torch.nn as nn

from options import Options

TRAJECTORY_PIONTS_LENGTH = 2

"""
Loss for MLP
"""
class Loss(nn.Module):
    def __init__(self, options):
        super(Loss, self).__init__()
        self.options = options

    def forward(self, prediction, inputs):
        """
        :param prediction: (batch_size, num_vehicles, horizon, 1)
        :param inputs: (batch_size, num_vehicles, 4)
        """
        num_vehicles, num_obstacles = self.options.vehicles[0], self.options.obstacles[0]
        N, _, horizon, _ = prediction.shape

        vehicle_encoding = 2 if self.options.trajectory.relative_frame_of_reference else 4
        vehicle_data = inputs[:, :vehicle_encoding*num_vehicles].reshape(N, num_vehicles, vehicle_encoding)
        obstacle_data = inputs[:, vehicle_encoding*num_vehicles:].reshape(N, num_obstacles, 3)
        vehicles = vehicle_data[:, :, :2]
        target = vehicle_data[:, :, 2:]
        obstacles = obstacle_data[:, :, :3]

        # convert prediction angles to vectors
        prediction = torch.cat((torch.cos(prediction), torch.sin(prediction)), dim=-1)

        trajectory_positions = torch.zeros((N, num_vehicles, horizon+1, 2))
        trajectory_positions[:, :, 0, :] = vehicles
        for i in range(horizon):
            trajectory_positions[:, :, i+1, :] = trajectory_positions[:, :, i, :] + (prediction[:, :, i, :] * TRAJECTORY_PIONTS_LENGTH)
        trajectory_positions = trajectory_positions[:, :, 1:, :]

        loss = 0

        # distance loss
        loss += torch.mean(torch.linalg.norm(target[:, :, None, :] - trajectory_positions, dim=-1)) * self.options.trajectory.loss.distance_cost

        # obstacle loss
        obstacles_distance = torch.linalg.norm(trajectory_positions[:, :, None, :, :] - obstacles[:, None, :, None, :2], dim=-1)
        obstacles_distance_near = obstacles_distance[obstacles_distance < (obstacles[:, None, :, None, 2] + self.options.trajectory.loss.obstacle_radius)]
        if len(obstacles_distance_near) > 0:
            loss += torch.mean(1/obstacles_distance_near) * self.options.trajectory.loss.obstacle_cost

        # vehicle loss
        vehicle_distance = torch.linalg.norm(trajectory_positions[:, 1:, :, :] - trajectory_positions[:, :-1, :, :], dim=-1)
        vehicle_distance_near = vehicle_distance[vehicle_distance < self.options.trajectory.loss.vehicle_radius]
        if len(vehicle_distance_near) > 0:
            loss += torch.mean(1/vehicle_distance_near) * self.options.trajectory.loss.vehicle_cost
        return loss
    

TRAJECTORY_PIONTS_LENGTH = 2
"""
Loss for GNN
"""
class GNN_Loss(nn.Module):
    """
    vehicles: (N, 5) [0, x, y, t_x, t_y]
    obstacles: (N, 5) [1, x, y, r, 0]
    """
    def __init__(self, options: Options) -> None:
        super().__init__()
        self.options = options
    
    def forward(self, X, prediction, edges_vehicles, edges_obstacles):
        """
        X: (N, 5): 
            if vehicle: [0, x, y, t_x, t_y]
            if obstacle: [1, x, y, r, 0]
        prediction: (N, horizon*classes)
        edges_vehicles: (2, M)
        edges_obstacles: (2, M)
        """
        # convert from classes to angle
        if self.options.trajectory.classes is not None:
            prediction = prediction.reshape(-1, self.options.trajectory.horizon, self.options.trajectory.classes)
            prediction_max_values, prediction_max_class  = torch.max(prediction, dim=-1)
            prediction = (prediction_max_class * (2 * np.pi / self.options.trajectory.classes)) * prediction_max_values

        prediction = torch.stack((torch.cos(prediction), torch.sin(prediction)),dim=-1)
        vehicle_mask = X[:, 0] == 0

        # each vehicle has a trajectory, obstacles with have zero everywhere but we need to keep track of there indices here.
        trajectory_positions = torch.zeros((len(X), self.options.trajectory.horizon+1, 2))
        trajectory_positions[vehicle_mask, 0, :] = X[vehicle_mask, 1:3]
        for i in range(self.options.trajectory.horizon):
            trajectory_positions[vehicle_mask, i+1, :] = trajectory_positions[vehicle_mask, i, :] + (prediction[vehicle_mask, i, :] * TRAJECTORY_PIONTS_LENGTH)
        trajectory_positions = trajectory_positions[:, 1:, :]

        # distance 
        distance_loss = torch.mean(torch.linalg.norm(X[vehicle_mask,None,3:5] - trajectory_positions[vehicle_mask], dim=-1)) * self.options.trajectory.loss.distance_cost

        # obstacle
        [from_obstacles, _]  = X[edges_obstacles]
        to_trajectory_positions = trajectory_positions[edges_obstacles[1]]
        obstacle_distance = torch.linalg.norm(from_obstacles[:,None,1:3] - to_trajectory_positions, dim=-1)
        obstacle_distance_near = obstacle_distance[obstacle_distance < (from_obstacles[:,None,3] + self.options.trajectory.loss.obstacle_radius)]
        if len(obstacle_distance_near) > 0:
            obstacle_loss = torch.mean(1 / obstacle_distance_near) * self.options.trajectory.loss.obstacle_cost
        else:
            obstacle_loss = torch.tensor(0.0)

        # penelize every horizon step to all horison steps of the other vehicle
        [from_trajectory_position, to_trajectory_position] = trajectory_positions[edges_vehicles]
        vehicle_distance = torch.linalg.norm(from_trajectory_position[:,:,None,:] - to_trajectory_position[:,None,:,:], dim=-1)
        vehicle_distance_near = vehicle_distance[vehicle_distance < self.options.trajectory.loss.vehicle_radius]
        if len(vehicle_distance_near) > 0:
            vehicle_loss = torch.mean(1 / vehicle_distance_near) * self.options.trajectory.loss.vehicle_cost
        else:
            vehicle_loss = torch.tensor(0.0)

        return distance_loss + obstacle_loss + vehicle_loss, distance_loss, obstacle_loss, vehicle_loss
