import torch
from vehicle.loss import Loss
import numpy as np

"""
This file algorithmically calculates the next target position for the vehicle and can be used instead of a learned approach.
"""

ADDITIONAL_OFFSET = 2
ADDITIONAL_VEHICLE_OFFSET = 4
STEP_SIZE = 2
STEP_SIZE_DIRECTION = 0.5
MAX_VELOCITY = 1.875

SLOW_DOWN_DISTANCE = 2.5

loss_function = Loss(None)

def get_direction(position: torch.Tensor, target: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
    """
    Get the direction to the target, taking into account the obstacles.
    :param position: The current position of the vehicle: shape (number_vehicles, 2)
    :param target: The target position of the vehicle: shape (number_vehicles, 2)
    :param obstacles: The position of the obstacles: shape (number_obstacles, 3)
    """
    # get normalized vector to target
    target_vector = target - position
    target_vector_normalized = target_vector / torch.linalg.norm(target_vector, dim=-1, keepdim=True)

    is_in_triangle = False
    for i in range(len(obstacles)):
        # check if this vector is in the triangle
        A1 = torch.stack((obstacles[i, 0], (obstacles[i, 1] + obstacles[i, 2] + ADDITIONAL_OFFSET)), dim=-1)
        A2 = torch.stack((obstacles[i, 0], (obstacles[i, 1] - obstacles[i, 2] - ADDITIONAL_OFFSET)), dim=-1)
        A3 = position

        B1 = torch.stack((obstacles[i, 0] + obstacles[i, 2] + ADDITIONAL_OFFSET, obstacles[i, 1]), dim=-1)
        B2 = torch.stack((obstacles[i, 0] - obstacles[i, 2] - ADDITIONAL_OFFSET, obstacles[i, 1]), dim=-1)
        B3 = position

        is_in_triangle |= \
            loss_function.point_in_triangle(position + target_vector_normalized, A1, A2, A3)
        is_in_triangle |= \
            loss_function.point_in_triangle(position + target_vector_normalized, B1, B2, B3)
    
    if not is_in_triangle:
        return target_vector_normalized * STEP_SIZE
    else:
        return _get_best_direction(position, target, obstacles) * STEP_SIZE
    
def get_angle(position: torch.tensor, next_position: torch.tensor):
    # get angle between current position and next position
    return torch.atan2(next_position[:, 1] - position[:, 1], next_position[:, 0] - position[:, 0])[:, None]

def get_velocity(position: torch.tensor, target: torch.tensor) -> torch.tensor:
    # function that depends on the distance to target and the current velocity
    current_distance_to_target = torch.linalg.norm(target - position[:, :2], dim=-1)
    current_velocity = position[:, 3]

    velocity = torch.zeros(len(position), 1)
    mask = current_distance_to_target > (SLOW_DOWN_DISTANCE * current_velocity)
    velocity[mask] = MAX_VELOCITY
    return velocity

    
    # if current_distance_to_target < (SLOW_DOWN_DISTANCE * current_velocity):
    #     return torch.tensor([[0]], dtype=torch.float32)
    # else:
    #     return torch.tensor([[MAX_VELOCITY]], dtype=torch.float32)

def get_next_target(position: torch.tensor, target: torch.tensor, obstacle: torch.tensor, vehicle_obstacles: torch.tensor) -> torch.tensor:
    if vehicle_obstacles != None:
        vehicle_obstacles = torch.cat((vehicle_obstacles, torch.ones(len(vehicle_obstacles), 1) * ADDITIONAL_VEHICLE_OFFSET), dim=1)
        obstacle = torch.concatenate((obstacle, vehicle_obstacles))
    next_target_position = get_direction_triangle_algorithm(position, target[:, :2], obstacle)
    # direction = position + torch.tensor(get_direction(position, target, obstacle), dtype=torch.float32)
    angle = get_angle(position, next_target_position)
    velocity = get_velocity(position, target[:, :2])

    if torch.linalg.norm(target[:, :2] - position[:, :2], dim=-1, keepdim=True) < STEP_SIZE_DIRECTION * 8:
        return torch.cat((target, torch.tensor([[0]], dtype=torch.float32)), dim=-1)
    return torch.cat((next_target_position, angle, velocity), dim=-1)

JUMP_LENGTH = 1
NUM_DIRECTIONS = 100
def _get_best_direction(position: torch.tensor, target: torch.tensor, obstacles: torch.tensor) -> torch.tensor:
    dx, dy = np.cos(np.linspace(0, 2*np.pi, NUM_DIRECTIONS)), np.sin(np.linspace(0, 2*np.pi, NUM_DIRECTIONS))
    dx *= JUMP_LENGTH
    dy *= JUMP_LENGTH
    possible_positions = position + np.stack((dx[None,:], dy[None,:]), axis=2).squeeze()
    possible_positions_mask = loss_function.points_in_triangle(position, possible_positions, obstacles)
    possible_positions_mask = torch.any(possible_positions_mask, dim=-1).squeeze()
    possible_positions = possible_positions[~possible_positions_mask]
    possible_positions_loss = torch.linalg.norm(possible_positions[:, None, :] - target[None, :, :], dim=-1, keepdim=True)
    vector = possible_positions[torch.argmin(possible_positions_loss)][None, :] - position
    return vector / torch.linalg.norm(vector, dim=-1, keepdim=True)


def get_direction_triangle_algorithm(position: torch.tensor, target: torch.tensor, obstacles: torch.tensor) -> torch.tensor:
    position = position[:, :2]
    if len(obstacles) == 0:
        direct_vectors = (target - position) / torch.linalg.norm(target - position, dim=-1, keepdim=True) * STEP_SIZE_DIRECTION
        return position + (direct_vectors / torch.linalg.norm(direct_vectors, dim=-1, keepdim=True) * STEP_SIZE)
    
    EPS = 0.0001
    A1 = torch.stack((obstacles[:, 0], (obstacles[:, 1] + obstacles[:, 2] + ADDITIONAL_OFFSET)), dim=-1)
    A1_updated = A1 + torch.tensor([0, EPS])
    A2 = torch.stack((obstacles[:, 0], (obstacles[:, 1] - obstacles[:, 2] - ADDITIONAL_OFFSET)), dim=-1)
    A2_updated = A2 + torch.tensor([0, -EPS])
    A3 = position

    B1 = torch.stack((obstacles[:, 0] + obstacles[:, 2] + ADDITIONAL_OFFSET, obstacles[:, 1]), dim=-1)
    B1_updated = B1 + torch.tensor([EPS, 0])
    B2 = torch.stack((obstacles[:, 0] - obstacles[:, 2] - ADDITIONAL_OFFSET, obstacles[:, 1]), dim=-1)
    B2_updated = B2 + torch.tensor([-EPS, 0])
    B3 = position

    A1_vectors = (A1_updated[None, :, :] - position[:, None, :]) / \
        torch.linalg.norm(A1_updated[None, :, :] - position[:, None, :], dim=-1, keepdim=True) * STEP_SIZE_DIRECTION
    A2_vectors = (A2_updated[None, :, :] - position[:, None, :]) / \
        torch.linalg.norm(A2_updated[None, :, :] - position[:, None, :], dim=-1, keepdim=True) * STEP_SIZE_DIRECTION

    B1_vectors = (B1_updated[None, :, :] - position[:, None, :]) / \
        torch.linalg.norm(B1_updated[None, :, :] - position[:, None, :], dim=-1, keepdim=True) * STEP_SIZE_DIRECTION
    B2_vectors = (B2_updated[None, :, :] - position[:, None, :]) / \
        torch.linalg.norm(B2_updated[None, :, :] - position[:, None, :], dim=-1, keepdim=True) * STEP_SIZE_DIRECTION
    
    direct_vectors = (target - position) / torch.linalg.norm(target - position, dim=-1, keepdim=True) * STEP_SIZE_DIRECTION
    triangle_based_vectors = torch.cat((A1_vectors, A2_vectors, B1_vectors, B2_vectors), dim=1)

    direct_vectors_position = position + direct_vectors
    triangle_based_vectors_position = position[:, None, :] + triangle_based_vectors

    # sort triangle_based_vectors_position by distance to target
    distances = torch.linalg.norm(triangle_based_vectors_position - target[:, None, :], dim=-1)
    distances_sorted_indices = torch.sort(distances, dim=1, descending=False)[1]
    triangle_based_vectors_position_sorted = triangle_based_vectors_position.squeeze(0)[distances_sorted_indices]

    all_positions = torch.cat((direct_vectors_position[:, None, :], triangle_based_vectors_position_sorted), dim=1)

    for i in range(all_positions.shape[1]):
        is_not_in_triangles = True
        for j in range(obstacles.shape[0]):
            is_not_in_triangles &= not torch.any(loss_function.point_in_triangle(all_positions[:, i, :],  A1, A2, A3))
            is_not_in_triangles &= not torch.any(loss_function.point_in_triangle(all_positions[:, i, :],  B1, B2, B3))
        if is_not_in_triangles:
            break

    if i == 0:
        return position + (direct_vectors / torch.linalg.norm(direct_vectors, dim=-1, keepdim=True) * STEP_SIZE)
    else:
        return position + ((all_positions[:, i, :] - position) / torch.linalg.norm(all_positions[:, i, :] - position, dim=-1, keepdim=True)) * STEP_SIZE

if __name__ == "__main__":
    position = torch.tensor([[20, 5]], dtype=torch.float)
    target = torch.tensor([[-20, 5]], dtype=torch.float)
    obstacle = torch.tensor([
        [0, 0, 4],
        [10, 10, 2],
        ], dtype=torch.float)
    print(get_next_target(position, target, obstacle))



    