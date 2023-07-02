import torch
import numpy as np
# from scripts.data_test import get_test_data_1, get_test_data_2, get_test_data_2_parking

# general 
# POSITION_RANGE = 15
OBSTACLE_SIZE_LOW = 1
OBSTACLE_SIZE_HIGH = 4

# normal
NORMAL_VELOCITY_MEAN = 2
NORMAL_VELOCITY_VARIANCE = 3
NORMAL_MIN_START_TARGET_DISTANCE = 10

# parking
PARKING_REFERENCE_VELOCITY_MEAN = 1.5
PARKING_REFERENCE_VELOCITY_VARIANCE = 3
PARKING_MIN_DISTANCE_MIN = 0
PARKING_MIN_DISTANCE_MAX = 7
PARKING_VELOCITY_MEAN = 2
PAKRING_VELOCITY_VARIANCE = 4
PARKING_RELATIVE_DISTANCE_MIN = 0
PARKING_RELATIVE_DISTANCE_MAX = 5

def get_data(length: int, number_vehicles: int, number_obstacles: int, normal_percentage: float, parking_percentage: float, behind_obstacle_percentage: float, is_start: bool) -> torch.Tensor:
    vehicles_normal, _ = _get_normal_data(int(length*normal_percentage), number_vehicles, number_obstacles, is_start)
    vehicles_parking, _ = get_parking_data(int(length*parking_percentage), number_vehicles, number_obstacles)
    vehicles_behind_obstacles, _ = _get_behind_obstacles_data(int(length*behind_obstacle_percentage), number_vehicles, number_obstacles)
    

    vehicles = np.concatenate((vehicles_normal, vehicles_parking, vehicles_behind_obstacles), axis=0)#.reshape(len(vehicles_normal) + len(vehicles_parking) + len(vehicles_behind_obstacles), 7)
    vehicles = torch.tensor(vehicles, dtype=torch.float32)

    if number_obstacles > 0:
        obstacles = _get_obstacles(starts=vehicles[:, :, :2], targets=vehicles[:, :, 4:6], num_obstacles=number_obstacles, is_start=is_start)
        obstacles_data = torch.zeros(len(vehicles), number_obstacles, 7)
        obstacles_data[:, :, :3] = torch.tensor(obstacles, dtype=torch.float32)
        obstacles_data = torch.tensor(obstacles_data, dtype=torch.float32)
        data = torch.cat((vehicles, obstacles_data), dim=1)
    else:
        data = torch.tensor(vehicles, dtype=torch.float32)

    # if len(data) == 1:
    #     data = data.repeat(2, 1)ob
    # data_normal = get_test_data_2(int(length * normal_percentage))
    # data_parking = get_test_data_2_parking(int(length * parking_percentage))
    # return torch.cat((data_normal, data_parking), dim=0)
    return data


def _get_normal_data(length: int, num_vehicles: int, num_obstacles: int, is_start=False) -> np.array:
    assert num_vehicles == 1 and (
        num_obstacles == 0 or num_obstacles == 1
    ), "This is not implemented yet"

    position_low_range = [-POSITION_RANGE, -POSITION_RANGE, -np.pi/2]
    position_high_range = [POSITION_RANGE, POSITION_RANGE, np.pi/2]
    position = np.random.uniform(
        low=position_low_range, high=position_high_range, size=(length, num_vehicles, 3)
    )
    velocity = np.random.normal(
        loc=NORMAL_VELOCITY_MEAN, scale=NORMAL_VELOCITY_VARIANCE, size=(length, num_vehicles, 1)
    )
    target = np.random.uniform(low=position_low_range, high=position_high_range, size=(length, num_vehicles, 3))

    if is_start:
        position = torch.tensor(position, dtype=torch.float32)
        indices = (np.linalg.norm(position[:, :, :2] - target[:,:,:2], axis=-1, ord=2) > 10)
        target = target[indices][:, None, :]
        position = torch.tensor([-20, 0, 0], dtype=torch.float32).repeat(len(target), num_vehicles, 1)
        velocity = torch.tensor([0], dtype=torch.float32).repeat(len(target), num_vehicles, 1)

    start = np.concatenate((position, velocity), axis=-1)
    vehicle = np.append(start, target, axis=2)

    return vehicle, torch.empty(len(vehicle), 1, 0)

def get_parking_data(length: int, num_vehicles: int, num_obstacles) -> torch.tensor:
    # low_range_pos = [-POSITION_RANGE, -POSITION_RANGE, -np.pi]
    # high_range_pos = [POSITION_RANGE, POSITION_RANGE, np.pi]
    # targets = np.random.uniform(low=low_range_pos, high=high_range_pos, size=(length, num_vehicles, 3))
    targets = torch.tensor([20, 0, 0]).repeat(length, num_vehicles, 1)

    reference_velocity = np.random.normal(loc=PARKING_REFERENCE_VELOCITY_MEAN, scale=PARKING_REFERENCE_VELOCITY_VARIANCE, size=(length))
    minimum_distance = np.random.uniform(low=PARKING_MIN_DISTANCE_MIN, high=PARKING_MIN_DISTANCE_MAX, size=(length))
    angle = np.random.uniform(low=-np.pi, high=np.pi, size=(length, num_vehicles, 1))
    velocity = np.random.normal(loc=PARKING_VELOCITY_MEAN, scale=PARKING_VELOCITY_MEAN, size=(length, num_vehicles, 1))

    position = np.ones((length, num_vehicles, 2))

    relative_angle = np.random.uniform(low=-np.pi, high=np.pi, size=(length, num_vehicles))
    relative_distance = np.random.uniform(low=PARKING_RELATIVE_DISTANCE_MIN, high=PARKING_RELATIVE_DISTANCE_MAX, size=(length, num_vehicles))
    arg_min = np.argmin(relative_distance, axis=-1)
    relative_distance[range(length),arg_min] = minimum_distance
    position[range(length),:,0] = targets[:,:,0]+relative_distance*np.cos(relative_angle)
    position[range(length),:,1] = targets[:,:,1]+relative_distance*np.sin(relative_angle)
    velocity[range(length),arg_min,0] = reference_velocity
    starts = np.concatenate((position,angle,velocity), axis=-1)
    vehicle = np.append(starts,targets,axis=2)
    return vehicle, torch.empty(len(vehicle), 1, 0)


OBSTACLE_IN_WAY_PERCENTAGE = 0.5
MIN_DISTANCE_FOR_OBSTACLE = 15
MIN_DISTANCE_TO_OBSTACLE = 3

def _get_obstacles(starts: torch.tensor, targets: torch.tensor, num_obstacles: int, is_start: bool) -> torch.tensor:
    low = [-POSITION_RANGE, -POSITION_RANGE, 2]
    high = [POSITION_RANGE, POSITION_RANGE, 4]
    obstacles = np.random.uniform(low=low, high=high, size=(len(starts), num_obstacles, 3))
    obstacles = torch.tensor(obstacles, dtype=torch.float32)
    return obstacles
    # obstacles = np.zeros((len(starts), num_obstacles, 3))
    # distance = torch.linalg.norm(targets - starts, dim=-1, ord=2)
    # v = (targets - starts)[distance > MIN_DISTANCE_FOR_OBSTACLE]
    # # normalize v
    # v = v / torch.linalg.norm(v, dim=-1, ord=2, keepdim=True)
    # # place obstacle randomly between target and start but not too close to either one

    # v = v * np.random.uniform(low=MIN_DISTANCE_TO_OBSTACLE, high=distance[distance > MIN_DISTANCE_FOR_OBSTACLE]-MIN_DISTANCE_TO_OBSTACLE, size=(len(v), 1))


    # v = targets[:, 0, :] - starts[:, 0, :]
    # v = v / (num_obstacles + 1)
    # for j in range(num_obstacles):    
    #     obstacles[:, j, :2] = (starts[:, 0, :] + (j+1) * v)
    #     # add randomness the the position by eps
    #     eps = np.random.normal(loc=0, scale=3, size=(len(starts), 2))
    #     obstacles[:, j, :2] += eps
    #     obstacles[:, j, 2] = np.random.uniform(low=0, high=4, size=(len(starts),))
    # return obstacles
    

    # check_point = np.concatenate((starts,targets),axis=0) 
    
    # v = targets - starts
    # d = np.linalg.norm(v, ord=2,axis=-1)
    
    # v = v/d[:,None]
    # h = v[:,[1,0]]
    # h[:,0] *= -1
    
    # obstacles = np.empty((0,3))
    
    # while(num_obstacles!= 0):
    
    #     h_shift1 = np.random.uniform(low=-7, high=0, size=(num_obstacles, len(d)))
    #     h_shift2 = np.random.uniform(low=0, high=7, size=(num_obstacles, len(d)))
    #     h_shift = np.concatenate((h_shift1,h_shift2),axis=0)
    #     v_shift = np.random.uniform(low=[-7]*len(d), high=(d+7).tolist(), size=(2*num_obstacles, len(d)))
    #     r = np.random.uniform(low=1, high=3, size=(2*num_obstacles*len(d),1))
        
    #     pos = (starts + v*v_shift[:,:,None] + h*h_shift[:,:,None]).reshape(-1,2)
        
    #     dist = np.linalg.norm(pos[:,None,:]-check_point, ord=2, axis=-1)
    #     min_dist = np.amin(dist, axis=-1)
    #     valid = (min_dist - r.flatten() -5) > 0
        
    #     candidates = np.concatenate((pos,r),axis=-1)[valid]
        
    #     if len(candidates) <= 0:
    #         continue
        
    #     n = min(len(candidates), num_obstacles)
        
    #     chosen = np.random.choice(range(len(candidates)), n, replace=False)           
    #     obstacles = np.concatenate((obstacles, candidates[chosen]), axis=0)
        
    #     num_obstacles -= n
    
    # return obstacles

def _get_behind_obstacles_data(length: int, num_vehicles: int, num_obstacles: int) -> torch.tensor:
    # [0, 0] is position of obstacles
    LOW_POSITION_RANGE = [-20, -2, -np.pi/2]
    HIGH_POSITION_RANGE = [-5, 2, np.pi/2]

    position = np.random.uniform(
        low=LOW_POSITION_RANGE, high=HIGH_POSITION_RANGE, size=(length, num_vehicles, 3)
    )
    velocity = np.random.normal(
        loc=NORMAL_VELOCITY_MEAN, scale=NORMAL_VELOCITY_VARIANCE, size=(length, num_vehicles, 1)
    )
    target = torch.tensor([20, 0, 0]).repeat(length, num_vehicles, 1)

    start = np.concatenate((position, velocity), axis=-1)
    vehicle = np.append(start, target, axis=2)

    return vehicle, torch.empty(len(vehicle), 1, 0)

CLOSE_MIN_DISTANCE = 2
CLOSE_MAX_DISTANCE = 4
START_VELOCITY_LOW = -2
TARGET_VELOCITY_LOW = 0
TARGET_VELOCITY_HIGH = 4
START_VELOCITY_HIGH = 4
START_TARGET_ANGLE_OFFSET = np.pi
TARGET_ANGLE_OFFSET = np.pi/4
POSITION_RANGE = 20

def get_close_data(length: int) -> torch.tensor:
    PARKING_PERCENTAGE = 0.0
    close_data_length = int(length * (1 - PARKING_PERCENTAGE))


    position_low_range = [-POSITION_RANGE, -POSITION_RANGE]
    position_high_range = [POSITION_RANGE, POSITION_RANGE]
    position = np.random.uniform(
        low=position_low_range, high=position_high_range, size=(close_data_length, 1, 2)
    )

    # get 0.25 of the starts with zero velocity
    velocity = np.zeros((close_data_length//4, 1, 1))    
    velocity = np.append(velocity, np.random.uniform(low=START_VELOCITY_LOW, high=START_VELOCITY_HIGH, size=(close_data_length - len(velocity), 1, 1)), axis=0)

    angle = np.random.uniform(low=-np.pi, high=np.pi, size=(close_data_length, 1, 1))
    vehicle = np.concatenate((position, angle, velocity), axis=-1)

    # get random anlge offset
    angle_offset = np.random.uniform(low=-START_TARGET_ANGLE_OFFSET/2, high=START_TARGET_ANGLE_OFFSET/2, size=(close_data_length, 1, 1))
    relative_target_angle = angle + angle_offset

    close_distance_factor = np.random.uniform(low=CLOSE_MIN_DISTANCE, high=CLOSE_MAX_DISTANCE, size=(close_data_length, 1, 1))
    # get target_position
    target_position = position[:, :, :2] + \
        close_distance_factor * np.concatenate((np.cos(relative_target_angle), np.sin(relative_target_angle)), axis=-1)
    
    target_angle = relative_target_angle + np.random.uniform(low=-TARGET_ANGLE_OFFSET/2, high=TARGET_ANGLE_OFFSET/2, size=(close_data_length, 1, 1))
    target_velocity = np.random.uniform(low=TARGET_VELOCITY_LOW, high=TARGET_VELOCITY_HIGH, size=(close_data_length, 1, 1))
    target = np.concatenate((target_position, target_angle, target_velocity), axis=-1)
    
    vehicle = vehicle.squeeze()
    target = target.squeeze()
    close_data = torch.tensor(np.concatenate((vehicle, target), axis=-1), dtype=torch.float32)

    parking_data = torch.tensor(get_parking_data(int(len(vehicle) * PARKING_PERCENTAGE), 1, 0)[0], dtype=torch.float32)
    parking_data = torch.cat((parking_data, torch.zeros(len(parking_data), 1, 1)), dim=-1).squeeze()
    return torch.cat((close_data, parking_data), dim=0)

MIN_DISTANCE = 40
EPS_ANGLE = np.pi/8
def get_inference_data(length: int, number_of_vehicles:int, number_of_obstacles: int) -> torch.tensor:
    region_to_range = {
        # upper right
        0: {"low": [POSITION_RANGE/2, POSITION_RANGE/2], "high": [POSITION_RANGE, POSITION_RANGE]},
        # upper left
        1: {"low": [-POSITION_RANGE, POSITION_RANGE/2], "high": [-POSITION_RANGE/2, POSITION_RANGE]},
        # lower left 
        2: {"low": [-POSITION_RANGE, -POSITION_RANGE], "high": [-POSITION_RANGE/2, -POSITION_RANGE/2]},
        # lower right
        3: {"low": [POSITION_RANGE/2, -POSITION_RANGE], "high": [POSITION_RANGE, -POSITION_RANGE/2]},
    }
    start_region_to_target_region = {
        0: 2,
        1: 3,
        2: 0,
        3: 1,
    }
    
    data_vehicles = np.zeros((length, number_of_vehicles, 7))
    data_obstacles = np.zeros((length, number_of_obstacles, 3))
    for i in range(length):
        random_set = {0, 1, 2, 3}
        vehicles = np.zeros((number_of_vehicles, 7))
        for j in range(number_of_vehicles):
            # get random int between 0 and 3
            region_index = np.random.randint(low=0, high=len(random_set), size=1).item()
            region = list(random_set)[region_index]
            random_set.remove(region)
            start = np.random.uniform(low=region_to_range[region]["low"], high=region_to_range[region]["high"], size=2)
            target = np.random.uniform(low=region_to_range[start_region_to_target_region[region]]["low"], high=region_to_range[start_region_to_target_region[region]]["high"], size=2)
            angle_start_target = np.arctan2(target[1] - start[1], target[0] - start[0])
            start_angle = angle_start_target + np.random.uniform(low=-EPS_ANGLE, high=EPS_ANGLE, size=1)
            target_angle = angle_start_target + np.random.uniform(low=-EPS_ANGLE, high=EPS_ANGLE, size=1)
            start = np.concatenate((start, start_angle), axis=-1)
            target = np.concatenate((target, target_angle), axis=-1)
            start = np.concatenate((start, np.zeros(1)), axis=-1)
            vehicles[j] = np.concatenate((start, target), axis=-1)
        
        obstacles = np.zeros((number_of_obstacles, 3))
        v = vehicles[0, 4:6] - vehicles[0, :2]
        v = v / (number_of_obstacles + 1)
        for j in range(number_of_obstacles):    
            obstacles[j, :2] = (vehicles[0, :2] + (j+1) * v)
            # add randomness the the position by eps
            eps = np.random.normal(loc=0, scale=3, size=2)
            obstacles[j, :2] += eps
            obstacles[j, 2] = np.random.uniform(low=2, high=4, size=1)
    
        data_vehicles[i] = vehicles
        data_obstacles[i] = obstacles

    return torch.tensor(data_vehicles, dtype=torch.float32), torch.tensor(data_obstacles, dtype=torch.float32)
    #     obstacles = obstacles.reshape(-1, 1, 3*number_of_obstacles)

    # # get angle of start to target vector
    # angle_start_target = np.random.uniform(low=-np.pi, high=np.pi, size=(length, number_of_vehicles, 1))
    # targets = starts[:, :, :2] + MIN_DISTANCE * np.concatenate((np.cos(angle_start_target), np.sin(angle_start_target)), axis=-1)

    # # add start angle
    # starts_angle = angle_start_target + np.random.uniform(low=-EPS_ANGLE, high=EPS_ANGLE, size=(length, number_of_vehicles, 1))
    # starts = np.concatenate((starts, starts_angle), axis=-1)

    # # add target angle
    # targets_angle = angle_start_target + np.random.uniform(low=-EPS_ANGLE, high=EPS_ANGLE, size=(length, number_of_vehicles, 1))
    # targets = np.concatenate((targets, targets_angle), axis=-1)

    # # add velocity to starts
    # starts = np.concatenate((starts, np.zeros((length, number_of_vehicles, 1))), axis=-1)

    # obstacles = None
    # if number_of_obstacles > 0:
    #     v = targets[:, :, :2] - starts[:, :, :2]
    #     v = v / (number_of_obstacles + 1)
    #     obstacles = np.zeros((len(starts), number_of_obstacles, 3))
    #     for i in range(number_of_obstacles):
    #         obstacles[:, i, :2] = (starts[:, 0, :2] + (i+1) * v[:, 0, :])
    #         # add randomness the the position by eps
    #         eps = np.random.normal(loc=0, scale=3, size=(len(starts), 2))
    #         obstacles[:, i, :2] += eps
    #         obstacles[:, i, 2] = np.random.uniform(low=0, high=4, size=(len(starts)))
    #     obstacles = obstacles.reshape(-1, 1, 3*number_of_obstacles)

    # return [starts, targets, obstacles]

# ADDITIONAL_OFFSET = 3
# def get_data_new(length: int) -> torch.tensor:
#     obstacles = np.random.normal(loc=[0, 0], scale=[7, 7], size=(length, 1, 2))
#     obstacles = np.concatenate((obstacles, np.random.uniform(low=2, high=4, size=(length, 1, 1))), axis=-1)

#     x = np.random.uniform(-POSITION_RANGE, POSITION_RANGE, (length, 1, 1))
#     y = np.random.uniform(-POSITION_RANGE, POSITION_RANGE, (length, 1, 1))
#     mask = (np.abs(x) < 15) | (np.abs(y) < 15)
#     while np.sum(mask) > 0:
#         x[mask] = np.random.uniform(-POSITION_RANGE, POSITION_RANGE, np.sum(mask))
#         y[mask] = np.random.uniform(-POSITION_RANGE, POSITION_RANGE, np.sum(mask))
#         mask = (np.abs(x) < 15) | (np.abs(y) < 15)
#     target = np.concatenate([x,y], axis=2)
#     target = np.concatenate([target, np.random.uniform(-np.pi, np.pi, (length, 1, 1))], axis=2)

#     points = np.random.normal(loc=[0, 0], scale=[10, 10], size=(length, 1, 2))
#     distances = np.linalg.norm(points - obstacles[:, :, :2], axis=-1) - obstacles[:, :, 2] - ADDITIONAL_OFFSET
#     mask = distances < 0
#     while np.sum(mask) > 0:
#         points[mask] = np.random.normal(loc=[0, 0], scale=[10, 10], size=(np.sum(mask), 2))
#         distances = np.linalg.norm(points - obstacles[:, :, :2], axis=-1) - obstacles[:, :, 2] - ADDITIONAL_OFFSET
#         mask = distances < 0
    
#     starts = np.concatenate((points, np.zeros((length, 1, 1)), np.random.uniform(-np.pi, np.pi, (length, 1, 1))), axis=2)
#     vehicles = np.concatenate((starts, target), axis=2)
#     obstacles = np.concatenate((obstacles, np.zeros((length, 1, 4))), axis=2)
#     return torch.tensor(np.concatenate((vehicles, obstacles), axis=1), dtype=torch.float32)
    
    
    # obstacles = np.random.normal(loc=)

BEHIND_OBSTACLE_PERCENTAGE = 0.5
NORMAL_PERCENTAGE = 0.5
def get_data_new(length: int, number_obstacles: int) -> torch.tensor:
    x = np.random.uniform(-POSITION_RANGE, POSITION_RANGE, (length, 1, 1))
    y = np.random.uniform(-POSITION_RANGE, POSITION_RANGE, (length, 1, 1))
    mask = (np.abs(x) < 15) | (np.abs(y) < 15)
    while np.sum(mask) > 0:
        x[mask] = np.random.uniform(-POSITION_RANGE, POSITION_RANGE, np.sum(mask))
        y[mask] = np.random.uniform(-POSITION_RANGE, POSITION_RANGE, np.sum(mask))
        mask = (np.abs(x) < 15) | (np.abs(y) < 15)
    target = np.concatenate([x,y], axis=2)
    target = np.concatenate([target, np.random.uniform(-np.pi, np.pi, (length, 1, 1))], axis=2)




    if number_obstacles > 0:
        obstacles = np.random.normal(loc=[0, 0], scale=[7, 7], size=(length, 1, 2))
        obstacles = np.concatenate((obstacles, np.random.uniform(low=2, high=4, size=(length, 1, 1))), axis=-1)
        obstacles = np.concatenate((obstacles, np.zeros((length, 1, 4))), axis=2)
    else:
        obstacles = np.empty((0, 1, 7))
    
    index = int(length*BEHIND_OBSTACLE_PERCENTAGE)
    # starts_behind_obstacles = get_starts_behind_obstacles(obstacles[:index, 0, :], target[:index, 0, :])
    # starts_behind_obstacles = np.concatenate((starts_behind_obstacles, np.zeros((index, 1)), np.zeros((index, 1))), axis=1)[:, None, :]

    starts_around_obstacles = get_starts_around_obstacle(obstacles[:index, 0, :])
    starts_around_obstacles = np.concatenate((starts_around_obstacles, np.zeros((len(starts_around_obstacles), 1)), np.zeros((len(starts_around_obstacles), 1))), axis=1)[:, None, :]

    starts_normal = get_starts_normal(obstacles[index:, 0, :], target[index:, 0, :])
    starts_normal = np.concatenate((starts_normal, np.zeros((len(starts_normal), 1)), np.zeros((len(starts_normal), 1))), axis=1)[:, None, :]

    # starts = np.concatenate((starts_behind_obstacles, starts_normal), axis=0)
    starts = np.concatenate((starts_around_obstacles, starts_normal), axis=0)
    vehicles = np.concatenate((starts, target), axis=2)

    if number_obstacles > 0:
        return torch.tensor(np.concatenate((vehicles, obstacles), axis=1), dtype=torch.float32)
    else:
        return torch.tensor(vehicles, dtype=torch.float32)

ADDITIONAL_OFFSET = 3
VARIANCE = [4, 4]
def get_starts_behind_obstacles(obstacles: torch.tensor, target: torch.tensor) -> torch.tensor:
    length = target.shape[0]
    if len(obstacles) == 0:
        return np.random.normal(loc=[0,0], scale=VARIANCE, size=(length, 2))

    v = target[:, :2] - obstacles[:, :2]
    v_norm = v/np.linalg.norm(v, axis=-1, keepdims=True)
    starts_mean = obstacles[:, :2] - v_norm * (obstacles[:, 2][:,None] + ADDITIONAL_OFFSET)
    starts = np.random.normal(loc=starts_mean, scale=VARIANCE, size=(length, 2))
    for i in range(length):
        min_distance = obstacles[i][2] + 1
        while np.linalg.norm(starts[i] - obstacles[i][:2]) < min_distance:
            starts[i] = np.random.normal(loc=starts_mean[i], scale=VARIANCE)
    return starts


def get_starts_around_obstacle(obstacles: torch.tensor) -> torch.tensor:
    variance = [5, 5]
    length = obstacles.shape[0]
    starts = np.random.normal(loc=obstacles[:, :2], scale=variance, size=(length, 2))
    for i in range(length):
        min_distance = obstacles[i][2] + 2
        while np.linalg.norm(starts[i] - obstacles[i][:2]) < min_distance:
            starts[i] = np.random.normal(loc=obstacles[i][:2], scale=variance)
    return starts


SCALE = [15,15]
def get_starts_normal(obstacles: torch.tensor, target: torch.tensor) -> torch.tensor:
    length = target.shape[0]
    if len(obstacles) == 0:
        return np.random.normal(loc=[0, 0], scale=SCALE, size=(length, 2))
    
    starts = np.random.normal(loc=[0, 0], scale=SCALE, size=(length, 2))
    for i in range(length):
        min_distance = obstacles[i][2] + ADDITIONAL_OFFSET
        while np.linalg.norm(starts[i] - obstacles[i][:2]) < min_distance:
            starts[i] = np.random.normal(loc=[0, 0], scale=SCALE)
    return torch.tensor(starts)


    # starts = np.concatenate([x,y,np.zeros((length,1,1))],axis=2)
    # for i in range(length):
    #     for j in range(10):
    #         point = np.random.uniform(low=-POSITION_RANGE + obstacle_size + offset,
    #                                 high=POSITION_RANGE - obstacle_size - offset,
    #                                 size=(1,1))
    #         while np.min(np.linalg.norm(points[i,:,:2]-point,axis=1)) < obstacle_size + offset:
    #             point = np.random.uniform(low=-POSITION_RANGE + obstacle_size + offset,
    #                                     high=POSITION_RANGE - obstacle_size - offset,
    #                                     size=(1,1))
    #         points[i] = np.concatenate([points[i],point,np.zeros((1))],axis=0)
    # return torch.tensor(points), torch.tensor(obstacles), torch.tensor(target)

    # points = np.zeros((length, 1, 2))
    # distances = np.zeros((length,))
    # for i in range(length):
    #     while True:
    #         points[i] = np.random.normal(loc=[0, 0], scale=[10, 10], size=(1,2))
    #         distances[i] = np.linalg.norm(points[i,:] - target[i,:,:2]) + np.linalg.norm(points[i,:,np.newaxis] - x[i,:2][:,0,np.newaxis])
    #         if distances[i] > ADDITIONAL_OFFSET:
    #             break
    #     obstacles[i,:,0:2] = points[i]
    #     obstacles[i,:,2] = distances[i] - ADDITIONAL_OFFSET
    
    # starts = np.concatenate((points + ADDITIONAL_OFFSET * (target[:, :, :2] - points) / distances.reshape(-1,1,1), np.zeros((length, 1, 1)), np.zeros((length, 1, 1))), axis=2)
    # vehicles = np.concatenate((starts, target), axis=2)
    # obstacles = np.concatenate((obstacles,np.zeros((length, 1, 4))), axis=2)
    
    # return torch.tensor(np.concatenate((vehicles, obstacles), axis=1), dtype=torch.float32)

def get_test_data(length: int) -> torch.tensor:
    low = [-20, -20, 0, 0, -20, -20, 0] 
    high = [20, 20, 0, 0, 20, 20, 0]
    data = np.random.uniform(low=low, high=high, size=(length, 1, 7))
    return torch.tensor(data, dtype=torch.float32)

NORMAL_PERCENTAGE = 0.5
AROUND_OBSTACLE_PERCENTAGE = 0.5
OBSTACLE_RANDOMNESS = 5
AROUND_OBSTACLE_VARIANCE = 5
ADDITIONAL_OBSTACLE_DISTANCE = 2
NORMAL_START_VARIANCE = 15
POSITION_RANGE = 30

REGION_TO_RANGE = {
    # upper right
    0: {"low": [POSITION_RANGE/2, POSITION_RANGE/2], "high": [POSITION_RANGE, POSITION_RANGE]},
    # upper left
    1: {"low": [-POSITION_RANGE, POSITION_RANGE/2], "high": [-POSITION_RANGE/2, POSITION_RANGE]},
    # lower left 
    2: {"low": [-POSITION_RANGE, -POSITION_RANGE], "high": [-POSITION_RANGE/2, -POSITION_RANGE/2]},
    # lower right
    3: {"low": [POSITION_RANGE/2, -POSITION_RANGE], "high": [POSITION_RANGE, -POSITION_RANGE/2]},
}

START_REGION_TO_TARGET_REGION = {
    0: 2,
    1: 3,
    2: 0,
    3: 1,
}

def get_data_with_obstacles_one_vehicle(length: int, number_obstacle: int, number_vehicles: int, is_start=False) -> torch.tensor:
    data_vehicles = np.zeros((length, number_vehicles, 7))
    data_obstacles = np.zeros((length, number_obstacle, 3))

    for i in range(length):
        random_set = {0, 1, 2, 3}
        vehicles = np.zeros((number_vehicles, 7))
        for j in range(number_vehicles):
            # get random int between 0 and 3
            region_index = np.random.randint(low=0, high=len(random_set), size=1).item()
            region = list(random_set)[region_index]
            random_set.remove(region)
            imaginary_start = np.random.uniform(low=REGION_TO_RANGE[region]["low"], high=REGION_TO_RANGE[region]["high"], size=2)
            target = np.random.uniform(low=REGION_TO_RANGE[START_REGION_TO_TARGET_REGION[region]]["low"], high=REGION_TO_RANGE[START_REGION_TO_TARGET_REGION[region]]["high"], size=2)
            angle_start_target = np.arctan2(target[1] - imaginary_start[1], target[0] - imaginary_start[0])
            start_angle = angle_start_target + np.random.uniform(low=-EPS_ANGLE, high=EPS_ANGLE, size=1)
            target_angle = angle_start_target + np.random.uniform(low=-EPS_ANGLE, high=EPS_ANGLE, size=1)
            imaginary_start = np.concatenate((imaginary_start, start_angle), axis=-1)
            target = np.concatenate((target, target_angle), axis=-1)
            imaginary_start = np.concatenate((imaginary_start, np.zeros(1)), axis=-1)
            vehicles[j] = np.concatenate((imaginary_start, target), axis=-1)
        
        obstacles = np.zeros((number_obstacle, 3))
        start_to_target_vector = vehicles[:, 4:6] - vehicles[:, :2]
        for o in range(number_obstacle):
            eps = np.random.normal(loc=0, scale=OBSTACLE_RANDOMNESS, size=2)
            obstacles[o, :2] = vehicles[0, :2] + (o+1) * start_to_target_vector[0] / (number_obstacle + 1) + eps
            obstacles[o, 2] = np.random.uniform(low=2, high=6, size=1)
        data_vehicles[i] = vehicles
        data_obstacles[i] = obstacles

    if is_start:
        obstacles = np.zeros((length, number_obstacle, 7))
        obstacles[:, :, :3] = data_obstacles
        data = np.concatenate((data_vehicles, obstacles), axis=1)
        return torch.tensor(data, dtype=torch.float32)

    around_obstacle_index = length * AROUND_OBSTACLE_PERCENTAGE
    actual_starts = np.zeros((length, number_vehicles, 4))
    # around obstacles
    if number_obstacle == 0:
        actual_starts[:, :, :2] = np.random.uniform(low=[-POSITION_RANGE, -POSITION_RANGE], high=[POSITION_RANGE, POSITION_RANGE], size=(length, number_vehicles, 2))
    else:
        for i in range(int(around_obstacle_index)):
            obstacles = data_obstacles[i]
            obstacle_index = np.random.randint(low=0, high=number_obstacle, size=1).item()
            start = np.random.normal(loc=obstacles[obstacle_index, :2], scale=AROUND_OBSTACLE_VARIANCE, size=2)
            for o in range(number_obstacle):
                min_distance = obstacles[o, 2] + ADDITIONAL_OBSTACLE_DISTANCE
                while np.linalg.norm(start - obstacles[o, :2]) < min_distance:
                    start = np.random.normal(loc=obstacles[o, :2], scale=AROUND_OBSTACLE_VARIANCE, size=2)
            actual_starts[i] = np.concatenate((start, np.zeros(2)), axis=-1)
        
        # normal starts
        for i in range(int(around_obstacle_index), length):
            start = np.random.uniform(low=[-POSITION_RANGE, -POSITION_RANGE], high=[POSITION_RANGE, POSITION_RANGE], size=2)
            # start = np.random.normal(loc=[0, 0], scale=SCALE, size=2)
            for o in range(number_obstacle):
                min_distance = obstacles[o, 2] + ADDITIONAL_OFFSET
                while np.linalg.norm(start - obstacles[o, :2]) < min_distance:
                    start = np.random.uniform(low=[-POSITION_RANGE, -POSITION_RANGE], high=[POSITION_RANGE, POSITION_RANGE], size=2)
            actual_starts[i] = np.concatenate((start, np.zeros(2)), axis=-1)
    
    vehicles = np.concatenate((actual_starts, data_vehicles[:, :, 4:7]), axis=-1)
    obstacles = np.zeros((length, number_obstacle, 7))
    obstacles[:, :, :3] = data_obstacles
    data = np.concatenate((vehicles, obstacles), axis=1)
    return torch.tensor(data, dtype=torch.float32)
            
        
def get_data_multiple_vehicle(length: int, number_vehicle: int) -> torch.tensor:
    data = np.zeros((length, number_vehicle, 7))
    for i in range(length):
        random_set = {0, 1, 2, 3}
        vehicles = np.zeros((number_vehicle, 7))
        for j in range(number_vehicle):
            # get random int between 0 and 3
            region_index = np.random.randint(low=0, high=len(random_set), size=1).item()
            region = list(random_set)[region_index]
            random_set.remove(region)
            start = np.random.uniform(low=REGION_TO_RANGE[region]["low"], high=REGION_TO_RANGE[region]["high"], size=2)
            target = np.random.uniform(low=REGION_TO_RANGE[START_REGION_TO_TARGET_REGION[region]]["low"], high=REGION_TO_RANGE[START_REGION_TO_TARGET_REGION[region]]["high"], size=2)
            angle_start_target = np.arctan2(target[1] - start[1], target[0] - start[0])
            start_angle = angle_start_target + np.random.uniform(low=-EPS_ANGLE, high=EPS_ANGLE, size=1)
            target_angle = angle_start_target + np.random.uniform(low=-EPS_ANGLE, high=EPS_ANGLE, size=1)
            start = np.concatenate((start, start_angle), axis=-1)
            target = np.concatenate((target, target_angle), axis=-1)
            start = np.concatenate((start, np.zeros(1)), axis=-1)
            vehicles[j] = np.concatenate((start, target), axis=-1)
        data[i] = vehicles
    return torch.tensor(data, dtype=torch.float32)

def get_data_two_vehicles(length: int, min_distance=3) -> torch.tensor:
    data = np.zeros((length, 2, 7))
    
    for i in range(length):
        while True:
            data[i, 0, :2] = np.random.uniform(low=[-20, 0], high=[20, 0], size=(1, 2))
            data[i, 1, :2] = np.random.uniform(low=[-20, 0], high=[20, 0], size=(1, 2))
            
            distance_start = np.linalg.norm(data[i, 0, :2] - data[i, 1, :2])
            distance_target_0 = np.linalg.norm(data[i, 0, :2] - np.array([20, 0]))
            distance_target_1 = np.linalg.norm(data[i, 1, :2] - np.array([-20, 0]))
            
            if (distance_start >= min_distance and
                distance_target_0 >= min_distance and
                distance_target_1 >= min_distance):
                break
                
    data[:, 0, 4:6] = np.repeat(np.array([[20, 0]]), length, axis=0)
    data[:, 1, 4:6] = np.repeat(np.array([[-20, 0]]), length, axis=0)

    return torch.tensor(data, dtype=torch.float32)

    # low_range = np.array([-POSITION_RANGE, -20])
    # high_range = np.array([POSITION_RANGE, 20])
    # starts = np.random.uniform(low=low_range, high=high_range, size=(length, 2, 2))
    
    # first_target = np.random.normal(loc=[-15, 0], scale=3, size=(length, 2))
    # second_target = np.random.normal(loc=[15, 0], scale=3, size=(length, 2))
    # data[:, :, :2] = starts
    # data[:, 0, 4:6] = first_target
    # data[:, 1, 4:6] = second_target
    # return torch.tensor(data, dtype=torch.float32)

def get_data_two_vehicles(length: int) -> torch.tensor:
    data = np.zeros((length, 2, 7))
    starts = np.random.uniform(low=[-POSITION_RANGE, -POSITION_RANGE], high=[POSITION_RANGE, POSITION_RANGE], size=(length, 2, 2))
    data[:, :, :2] = starts
    targets = np.random.uniform(low=[-POSITION_RANGE, -POSITION_RANGE], high=[POSITION_RANGE, POSITION_RANGE], size=(length, 2, 2))
    data[:, :, 4:6] = targets
    return torch.tensor(data, dtype=torch.float32)

def REGION_TO_NUMBER(x, y):
    # upper right
    if x > 0 and y > 0:
        return 0
    # upper left
    elif x < 0 and y > 0:
        return 1
    # lower left
    elif x < 0 and y < 0:
        return 2
    # lower right
    elif x > 0 and y < 0:
        return 3
    else:
        raise ValueError("x and y must be non-zero")

COLLISION_POSITION_RANGE = 15
MEAN_DISTANCE_VEHICLES = 4
VARIANCE_DISTANCE_VEHICLES = 0.5
def get_data_two_vehicles_collision(length: int, number_vehicle: int, min_distance=3) -> torch.tensor:
    data = np.zeros((length, number_vehicle, 7))
    for i in range(length):
        start = np.random.uniform(low=[-COLLISION_POSITION_RANGE, -COLLISION_POSITION_RANGE], high=[COLLISION_POSITION_RANGE, COLLISION_POSITION_RANGE], size=(1, 2))
        # get region
        start_region = REGION_TO_NUMBER(start[0, 0], start[0, 1])
        target_region = START_REGION_TO_TARGET_REGION[start_region]
        target = np.random.uniform(low=REGION_TO_RANGE[target_region]["low"], high=REGION_TO_RANGE[target_region]["high"], size=(1, 2))
        data[i][0, :2] = start
        data[i][0, 4:6] = target

        vector_start_target = target - start
        vector_start_target = vector_start_target / np.linalg.norm(vector_start_target)
        second_start_mean = start + vector_start_target * MEAN_DISTANCE_VEHICLES
        start_second = np.random.normal(loc=second_start_mean, scale=VARIANCE_DISTANCE_VEHICLES, size=(1, 2))

        # get vector between the two starts
        vector = start - start_second
        while np.linalg.norm(vector) < min_distance:
            start_second = np.random.normal(loc=second_start_mean, scale=VARIANCE_DISTANCE_VEHICLES, size=(1, 2))
            vector = start - start_second
        target_second = start_second + vector * np.random.uniform(low=4, high=10, size=(1, 2))
        data[i][1, :2] = start_second
        data[i][1, 4:6] = target_second
    return torch.tensor(data, dtype=torch.float32)

NORMAL_PERCENTAGE = 0.5
COLLISION_PRECENTAGE = 0.5
def get_two_vehicles_data(length: int) -> torch.tensor:
    data = np.zeros((length, 2, 7))
    normal_length = int(length * NORMAL_PERCENTAGE)
    collision_length = length - normal_length
    data[:normal_length] = get_data_two_vehicles_new(normal_length)
    data[normal_length:] = get_data_two_vehicles_collision(collision_length, 2)
    return torch.tensor(data, dtype=torch.float32)

def get_two_vehicles_inference(length: int) -> torch.tensor:
    data = np.zeros((length, 2, 7))
    for i in range(length):
        random_set = {0, 1, 2, 3}
        region_index = np.random.randint(low=0, high=len(random_set), size=1).item()
        region = list(random_set)[region_index]
        start = np.random.uniform(low=REGION_TO_RANGE[region]["low"], high=REGION_TO_RANGE[region]["high"], size=2)
        target = np.random.uniform(low=REGION_TO_RANGE[START_REGION_TO_TARGET_REGION[region]]["low"], high=REGION_TO_RANGE[START_REGION_TO_TARGET_REGION[region]]["high"], size=2)
        # get angle of start and target
        angle = np.arctan2(target[1] - start[1], target[0] - start[0])
        data[i][0, :2] = start
        data[i][0, 4:6] = target
        data[i][0, 2] = angle
        data[i][1, 2] = angle + np.pi
        data[i][1, :2] = target
        data[i][1, 4:6] = start
    return torch.tensor(data, dtype=torch.float32)

        


RANGE_OFFSET = 5
ADDITIONAL_VARIANCE = 2
def get_data_two_vehicles_new(length: int) -> torch.tensor:
    data = np.zeros((length, 2, 7))
    for i in range(length):
        random_set = {0, 1, 2, 3}
        for j in range(2):
            region_index = np.random.randint(low=0, high=len(random_set), size=1).item()
            region = list(random_set)[region_index]
            start = np.random.uniform(low=REGION_TO_RANGE[region]["low"], high=REGION_TO_RANGE[region]["high"], size=2)
            target = np.random.uniform(low=REGION_TO_RANGE[START_REGION_TO_TARGET_REGION[region]]["low"], high=REGION_TO_RANGE[START_REGION_TO_TARGET_REGION[region]]["high"], size=2)
            
            # start to target vector
            vector = target - start
            vector_norm = vector / np.linalg.norm(vector)
            range_vector = (target + RANGE_OFFSET * vector_norm - (start - RANGE_OFFSET * vector_norm))

            random_point = np.random.uniform(low=0, high=1, size=1).item()
            random_point = (start - RANGE_OFFSET * vector_norm) + random_point * range_vector
            random_point += np.random.normal(loc=0, scale=ADDITIONAL_VARIANCE, size=2)
            data[i][j, :2] = random_point
            data[i][j, 4:6] = target
    return torch.tensor(data, dtype=torch.float32)


NORMAL_VARIANCE = 20
def generate_points_inside_rectangle(p1, p2, p3, p4, length, mid):
    x_min = min(p1[0], p2[0], p3[0], p4[0])
    x_max = max(p1[0], p2[0], p3[0], p4[0])
    y_min = min(p1[1], p2[1], p3[1], p4[1])
    y_max = max(p1[1], p2[1], p3[1], p4[1])

    points = []
    while len(points) < length:
        x = np.random.normal(mid[0], NORMAL_VARIANCE)
        y = np.random.normal(mid[1], NORMAL_VARIANCE)

        if x_min <= x <= x_max and y_min <= y <= y_max:
            points.append((x, y))

    return points