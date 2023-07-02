from torch.utils.data import Dataset, DataLoader
import torch

class MLP_Dataset_Trajectory(Dataset):
    def __init__(self, 
                 data: torch.tensor,
                 number_of_vehicle: None,
                 number_of_obstacles: None,
                 relative_frame_of_reference: False
                ) -> None:
        super().__init__()
        vehicles = data[:, :number_of_vehicle, [0, 1, 4, 5]]
        obstacles = data[:, number_of_vehicle:, [0, 1, 2]]
        self.data = torch.cat((vehicles.reshape(-1, 4*number_of_vehicle), obstacles.reshape(-1, 3*number_of_obstacles)), dim=1)
        self.relative_frame_of_reference = relative_frame_of_reference
        if relative_frame_of_reference:
            self.relative_data = change_to_relative_frame_of_reference_trajectory(vehicles, obstacles)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> torch.tensor:
        if self.relative_frame_of_reference:
            return self.data[index], self.relative_data[index]
        return self.data[index], None

def collect_fn(data: list) -> tuple:
    x = torch.stack([item[0] for item in data])
    if data[0][1] is None:
        return x, None
    y = torch.stack([item[1] for item in data])
    return x, y

class MLP_Dataloader(DataLoader):
    def __init__(
        self, dataset: MLP_Dataset_Trajectory, batch_size: int, shuffle: bool, drop_last: bool
    ):
        super().__init__(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collect_fn
        )

def change_to_relative_frame_of_reference_trajectory(vehicles: torch.tensor, obstacles: torch.tensor) -> torch.tensor:
    num_vehicles = vehicles.shape[1]
    num_obstacles = obstacles.shape[1]
    relative_vehicles = torch.zeros((len(vehicles), vehicles.shape[1], 2*num_vehicles))
    relative_vehicles[:, :, :2] = vehicles[:,:,2:] - vehicles[:,:,:2]
    for i in range(num_vehicles):
        done_vehicle_index = 1
        for j in range(num_vehicles):
            if i != j:
                relative_vehicles[:, i, 2*done_vehicle_index:2*(done_vehicle_index+1)] = vehicles[:,j,:2] - vehicles[:,i,:2]
                done_vehicle_index += 1
    if obstacles.shape[1] == 0:
        return relative_vehicles
    relative_obstacles = torch.zeros((len(obstacles), obstacles.shape[1], 3))
    relative_obstacles[:,:,:2] = obstacles[:,:,:2] - vehicles[:,:,:2]
    relative_obstacles[:,:,2] = obstacles[:,:,2]
    return torch.cat((relative_vehicles.reshape(-1, 2*num_vehicles), relative_obstacles.reshape(-1,3*num_obstacles)), dim=1)
