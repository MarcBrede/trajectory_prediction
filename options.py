from dataclasses import dataclass
from typing import List  

@dataclass
class Options:
    # vehicles are a list of int
    vehicles: List[int]
    obstacles: List[int]
    base_directory: str
    trajectory: "SettingsTrajectory"
    vehicle: "SettingsVehicle"
    visualization: "SettingsVisualization"

@dataclass
class SettingsTrajectory:
    horizon: int
    data_length: int
    model_name: str
    hyperparams: "Hyperparams"
    loss: "LossTrajectory"
    relative_frame_of_reference: bool
    classes: int

@dataclass
class SettingsVehicle:
    model_name: str
    horizon: int
    data_length: int
    normal_percentage: float
    parking_percentage: float
    relative_frame_of_reference: bool
    hyperparams: "Hyperparams"
    loss: "LossVehicle"


@dataclass
class Hyperparams:
    epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    learning_rate_patience: int
    learning_rate_factor: float
    learning_rate_minimum: float
    weight_decay: float
    cool_down: int
@dataclass
class LossTrajectory:
    distance_cost: float
    obstacle_cost: float
    obstacle_radius: float
    vehicle_radius: float
    vehicle_cost: float

@dataclass
class LossVehicle:
    distance_cost: float
    angle_cost: float
    just_last_step: bool
    triangluar_loss: bool
    velocity_cost: float

@dataclass
class SettingsVisualization:
    figure_size: int
    figure_limit: int
    ticks_step: int
    car_size: "Size"
    show_plot: bool
    save_plot: bool
    duration: int

@dataclass
class Size:
    width: int
    length: int
