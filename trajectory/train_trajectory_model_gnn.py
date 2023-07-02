import os
import torch
import pprint
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import random
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_data_with_obstacles_one_vehicle
from options import Options, SettingsTrajectory, Hyperparams, LossTrajectory, SettingsVisualization, Size 
from trajectory.helper import convert_normal_data_to_gnn_data
from trajectory.dataset_gnn import GNN_Dataset_Trajectory, GNN_Dataloader
from gnn import GNNModel
from mlp import MLPModel
from trajectory.loss import GNN_Loss
from vehicle.train_vehicle_model import VEHICLE_SETTINGS
from vehicle.dataset import change_to_relative_frame_of_reference_vehicle
from visualization.simulation import sim_run

TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.2

PRE_TRAINED_MODEL = None

def train_trajectory_model(options: Options):
    model_path = os.path.join(options.base_directory, "trajectory_gnn", options.trajectory.model_name)
    path = Path(model_path)
    path.mkdir(parents=True, exist_ok=True)

    train_data, valid_data, train_loader, valid_loader, model, optimizer, scheduler, criterion = prepare_objects(options)
    
    model_path = os.path.join(model_path, "model.pth")

    ############################
    # model = train_model(options, criterion, train_loader, valid_loader, model, optimizer, scheduler)
    ############################
    model.load_state_dict(torch.load(model_path))

    return model, valid_data

def prepare_objects(options: Options):
    tmp_data_train = {}
    tmp_data_valid = {}
    number_of_vehicles = options.vehicles[0]
    for _, num_obstacles in enumerate(options.obstacles):
        tmp_data_train[(number_of_vehicles+num_obstacles, number_of_vehicles)] = \
            get_data_with_obstacles_one_vehicle(int(options.trajectory.data_length*TRAIN_SPLIT)+1, num_obstacles, number_of_vehicles)
        tmp_data_valid[(number_of_vehicles+num_obstacles, number_of_vehicles)] = \
            get_data_with_obstacles_one_vehicle(int(options.trajectory.data_length*VALID_SPLIT)+1, num_obstacles, number_of_vehicles)

    train_data_gnn = convert_normal_data_to_gnn_data(tmp_data_train)
    valid_data_gnn = convert_normal_data_to_gnn_data(tmp_data_valid)

    train_dataset = GNN_Dataset_Trajectory(train_data_gnn)
    valid_dataset = GNN_Dataset_Trajectory(valid_data_gnn)

    train_data = train_dataset.data
    valid_data = valid_dataset.data

    train_loader = GNN_Dataloader(train_dataset, options.trajectory.hyperparams.batch_size, shuffle=True, drop_last=True)
    valid_loader = GNN_Dataloader(valid_dataset, options.trajectory.hyperparams.batch_size, shuffle=False, drop_last=False)
    
    bound = [math.pi]
    assert (bound is not None) == (options.trajectory.classes is None), "bound and classes must be set accordingly"

    model = GNNModel(
        input_features=5,
        output_features=options.trajectory.horizon if options.trajectory.classes is None else options.trajectory.horizon*options.trajectory.classes,
        max_num_vehicles=number_of_vehicles,
        max_num_obstacles=max(options.obstacles),
        bound=bound
    )

    if PRE_TRAINED_MODEL is not None:
        model.load_state_dict(torch.load(os.path.join(options.base_directory, "trajectory_gnn", PRE_TRAINED_MODEL)))

    optimizer = Adam(
        model.parameters(), 
        lr = options.trajectory.hyperparams.learning_rate, 
        weight_decay = options.trajectory.hyperparams.weight_decay
    )

    scheduler = ReduceLROnPlateau(
        optimizer = optimizer, mode = 'min', verbose = True, 
        patience = options.trajectory.hyperparams.learning_rate_patience,
        factor = options.trajectory.hyperparams.learning_rate_factor, 
        min_lr = options.trajectory.hyperparams.learning_rate_minimum,
        cooldown=10
    )
    
    criterion = GNN_Loss(options=options)

    return train_data, valid_data, train_loader, valid_loader, model, optimizer, scheduler, criterion

def train_model(options: Options, cirterion, train_loader, valid_loader, model, optimizer, scheduler):
    model_path = os.path.join(options.base_directory, "trajectory_gnn", options.trajectory.model_name)
    plt.clf()
    with open(os.path.join(model_path, "options.txt"), 'w') as f:
        pprint.pprint(vars(options.trajectory), stream=f)

    LOGS = {
        "training_loss": [],
        "validation_loss": [],
    }
    best_loss = np.inf

    def validate():
        model.eval()
        loss_array = np.array([])
        with torch.no_grad():
            for X, n in valid_loader:
                pred, _, _, edges_vehicles, edges_obstacles, _ = model(X, n)
                loss, _, _, _ = cirterion(X, pred, edges_vehicles, edges_obstacles)
                loss_array = np.append(loss_array, loss.item())
        valid_loss = np.mean(loss_array)
        LOGS["validation_loss"].append(valid_loss)
        return valid_loss

    def train():
        model.train()
        loss_array = np.array([])
        for X, n in train_loader:
            optimizer.zero_grad()
            pred, _, _, edges_vehicles, edges_obstacles, _ = model(X, n)
            loss, _, _, _ = cirterion(X, pred, edges_vehicles, edges_obstacles)
            loss.backward()
            optimizer.step()
            loss_array = np.append(loss_array, loss.item())
        train_loss = np.mean(loss_array)
        LOGS["training_loss"].append(train_loss)
        return train_loss

    valid_loss = validate()
    print('Loss without training: {}'.format(valid_loss))
    LOGS['training_loss'].append(None)
    patience_f = options.trajectory.hyperparams.patience

    for epoch in tqdm(range(options.trajectory.hyperparams.epochs)):
        train_loss = train()
        valid_loss = validate()

        # Log the losses
        LOGS["training_loss"].append(train_loss)
        LOGS["validation_loss"].append(valid_loss)

        if epoch > options.trajectory.hyperparams.cool_down:
            scheduler.step(train_loss)

        msg = f'Epoch: {epoch+1}/{options.trajectory.hyperparams.epochs} | Train Loss: {train_loss:.6} | Valid Loss: {valid_loss:.6}'
        print(msg)

        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_f = options.trajectory.hyperparams.patience
            torch.save(model.state_dict(), os.path.join(model_path, "model.pth"))
            print('Model Saved!')

        elif epoch > options.trajectory.hyperparams.cool_down:
            patience_f -= 1
            if patience_f == 0:
                print(f'Early stopping (no improvement since {options.trajectory.hyperparams.patience} models) | Best Valid Loss: {best_loss:.6f}')
                break

    plt.plot(LOGS["training_loss"], label='Training Loss')
    plt.plot(LOGS["validation_loss"], label='Validation Loss')
    plt.text(len(LOGS["validation_loss"]) - 1, best_loss, f'Best Valid Loss: {best_loss:.4f}', fontsize=12, verticalalignment='center', horizontalalignment='right')
    plt.legend()
    plt.savefig(os.path.join(model_path, "learning.png"))
    plt.show()
    return model


if __name__ == "__main__":
    options = Options(
        vehicles=[1],
        obstacles=[0,1,2,3],
        base_directory='./output/',
        trajectory=SettingsTrajectory(
        model_name="base_model",
            horizon=1,
            data_length=100000,
            relative_frame_of_reference=False,
            hyperparams=Hyperparams(
                epochs=1000,
                patience=150,
                batch_size=512*4,
                learning_rate=1e-4,
                learning_rate_patience=300,
                learning_rate_factor=0.2,
                learning_rate_minimum=1e-10,
                weight_decay=1e-7,
                cool_down=50,
            ),
            loss=LossTrajectory(
                distance_cost=1.0,
                obstacle_cost=10.0,
                obstacle_radius=1.0,
                vehicle_cost=0.0,
                vehicle_radius=2.0,
            ),
            # set to True to convert to classification problem
            classes=None
        ),
        vehicle=VEHICLE_SETTINGS,
        visualization=SettingsVisualization(
            figure_size=8,
            figure_limit=35,
            ticks_step=5,
            car_size=Size(width=1.0, length=2.5),
            show_plot=False,
            save_plot=True,
            duration=100
        ),
    )

    # depending on where the file is executed
    options.base_directory = options.base_directory if os.getcwd().endswith('trajectory_prediction') else os.path.join('../', options.base_directory)

    # train the trajectory model
    model, valid_data = train_trajectory_model(options)

    # load the already trained vehicle model
    vehicle_model_name = "base_model"
    vehicle_model_path = os.path.join(options.base_directory, "vehicle", vehicle_model_name, "model.pth")
    input_size = 4 if options.vehicle.relative_frame_of_reference else 8
    output_size = 2 * options.vehicle.horizon
    vehicle_model = MLPModel(
        input_size=input_size,
        output_size=output_size,
        bound=torch.tensor([1, 0.8]),
    )
    vehicle_model.load_state_dict(torch.load(vehicle_model_path))

    # get data for inference
    INFERENCE_VEHILCES = [1, 2]
    INFERENCE_OBSACLES = [0, 1, 2, 3, 4]
    inference_data_dict = {}
    for _, number_of_vehicles in enumerate(INFERENCE_VEHILCES):
        for _, number_of_obstacles in enumerate(INFERENCE_OBSACLES):
            inference_data_dict[(number_of_vehicles, number_of_obstacles)] = \
                get_data_with_obstacles_one_vehicle(100, number_of_obstacles, number_of_vehicles, is_start=True)

    # visualization
    for i in range(100):
        number_of_vehicles, number_of_obstacles = random.choice(INFERENCE_VEHILCES), random.choice(INFERENCE_OBSACLES)
        inference_data = inference_data_dict[(number_of_vehicles, number_of_obstacles)]
        inference_data_vehicles = inference_data[:, :number_of_vehicles, :]
        inference_data_obstacles = inference_data[:, number_of_vehicles:, :]

        sim_run(
            inference_data_vehicles[i, :, :4].numpy(),
            inference_data_vehicles[i, :, 4:7].numpy(),
            inference_data_obstacles[i,:,:3].numpy(),
            options,
            model=vehicle_model,
            trajectory_model=model,
            relative_frame_of_reference_converter_vehicle=change_to_relative_frame_of_reference_vehicle,
            relative_frame_of_reference_converter_trajectory=None,
            is_gnn=True,
            plot_name=str(i),
        )