import torch
from torch.utils.data import Dataset, DataLoader

class MLP_Dataset_Vehicle(Dataset):
    def __init__(self, 
                 data: torch.tensor, 
                 relative_frame_of_reference: False,
                 number_of_vehicle: None,
                 number_of_obstacles: None) -> None:
        super().__init__()
        
        self.data = data
        self.relative_frame_of_reference = relative_frame_of_reference
        if relative_frame_of_reference:
            self.relative_data = change_to_relative_frame_of_reference_vehicle(
                data,
                number_of_vehicle,
                number_of_obstacles,
            )
        else: 
            self.relative_data = None
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> torch.tensor:
        if self.relative_frame_of_reference:
            return self.data[index], self.relative_data[index]
        else:
            return self.data[index], None


def _collate_fn(batch):
    return torch.stack([d[0] for d in batch]), None

class MLP_Dataloader_Vehicle(DataLoader):
    def __init__(
        self, dataset: MLP_Dataset_Vehicle, batch_size: int, shuffle: bool, drop_last: bool, relative_frame_of_reference: bool
    ):
        collate_fn = None if relative_frame_of_reference else _collate_fn
        super().__init__(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, 
            collate_fn= collate_fn,
        )

def change_to_relative_frame_of_reference_vehicle(data: torch.tensor, num_of_vehicles: int, num_of_obstacles: int) -> torch.tensor:
    N = len(data)
    vehicle_index = 8 
    vehicles = data[:, :vehicle_index]
    if num_of_obstacles>0:
        obstacles = data[:, vehicle_index:]

    # get inverse rotation matrix (A^-1 = A^T)
    rotation_matrix = torch.zeros(N, 2, 2)
    rotation_matrix[:, 0, 0] = torch.cos(vehicles[:, 2])
    rotation_matrix[:, 0, 1] = torch.sin(vehicles[:, 2])
    rotation_matrix[:, 1, 0] = -torch.sin(vehicles[:, 2])
    rotation_matrix[:, 1, 1] = torch.cos(vehicles[:, 2])

    # get relative target position and angle
    relative_target_position = vehicles[:, 4:7] - vehicles[:, :3]
    relative_target_position[:, :2] = torch.matmul(rotation_matrix, relative_target_position[:, :2].unsqueeze(-1)).squeeze(-1)

    # get relative obstacle position
    if num_of_obstacles>0:
        relative_obstacle_position = obstacles[:, :2] - vehicles[:, :2]
        relative_obstacle_position = torch.matmul(rotation_matrix, relative_obstacle_position.unsqueeze(-1)).squeeze(-1)
        relative_obstacle_position = torch.cat((relative_obstacle_position, obstacles[:, 2][:, None]), dim=-1)
    else:
        relative_obstacle_position = torch.empty(N, 0)
    
    targets = torch.cat((relative_target_position, vehicles[:, -1][:, None]), dim=-1)
    return torch.cat((targets, relative_obstacle_position), dim=1)