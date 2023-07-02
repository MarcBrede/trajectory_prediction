import torch
import numpy as np

"""
Takes data a dict with keys: (number_of_entities, number_of_vehicles) and values: X [length, number_of_entities, 7]
Returns a a list of dict with keys: (number_of_entities, number_of_vehicles) and values: (X [length, 4], batch [length/number_of_entities, 4])

vehicle: [0, x, y, t_x, t_y]
obstacles: [1, x, y, r, 0]
"""
def convert_normal_data_to_gnn_data(data) -> torch.tensor:
    for key, value in data.items():
        number_entities, number_vehicles = key
        X = value
        length = X.shape[0]

        vehicles = torch.zeros((X.shape[0], number_vehicles, 5))
        # position
        vehicles[:, :, 1:3] = X[:, :number_vehicles, [0, 1]]
        # target position
        vehicles[:, :, 3:5] = X[:, :number_vehicles, [4, 5]]

        obstacles = torch.ones((X.shape[0], number_entities-number_vehicles, 5))
        obstacles[:, :, 1:4] = X[:, number_vehicles:, [0, 1, 2]]
        X = torch.cat((vehicles, obstacles), dim=1).reshape(-1, 5)
        batch = torch.tensor([number_entities, number_vehicles])[None,:].repeat(length, 1)
        data[key] = (X, batch)
    return data