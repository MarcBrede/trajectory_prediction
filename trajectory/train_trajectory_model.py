import os
import sys
import torch
import math
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
import pprint
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options import Options, SettingsTrajectory, Hyperparams, LossTrajectory, Size, SettingsVisualization
from mlp import MLPModel
from trajectory.dataset_mlp import MLP_Dataset_Trajectory, MLP_Dataloader, change_to_relative_frame_of_reference_trajectory
from trajectory.loss import Loss
from vehicle.dataset import change_to_relative_frame_of_reference_vehicle
from data import get_data_with_obstacles_one_vehicle, get_data
from vehicle.train_vehicle_model import VEHICLE_SETTINGS

from visualization.simulation import sim_run


PRE_TRAINED = None

TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.2

def train_trajectory_model(options: Options):
    model_path = os.path.join(options.base_directory, "trajectory_mlp", options.trajectory.model_name)
    path = Path(model_path)
    path.mkdir(parents=True, exist_ok=True)

    train_data, valid_data, train_loader, valid_loader, model, optimizer, scheduler, criterion = prepare_objects(options)
    
    model_path = os.path.join(model_path, "model.pth")

    ############################
    # model = train_model(options, criterion, train_loader, valid_loader, model, optimizer, scheduler)
    ###########################

    return model, valid_data

def prepare_objects(options: Options):
    number_of_vehicles = options.vehicles[0]
    number_of_obstacles = options.obstacles[0]

    train_data = get_data_with_obstacles_one_vehicle(int(options.trajectory.data_length*TRAIN_SPLIT)+1, number_of_obstacles, number_of_vehicles)
    valid_data = get_data_with_obstacles_one_vehicle(int(options.trajectory.data_length*VALID_SPLIT)+1, number_of_obstacles, number_of_vehicles)

    train_dataset = MLP_Dataset_Trajectory(
        data=train_data,
        number_of_vehicle=number_of_vehicles,
        number_of_obstacles=number_of_obstacles,
        relative_frame_of_reference=options.trajectory.relative_frame_of_reference,
    )

    valid_dataset = MLP_Dataset_Trajectory(
        data=train_data,
        number_of_vehicle=number_of_vehicles,
        number_of_obstacles=number_of_obstacles,
        relative_frame_of_reference=options.trajectory.relative_frame_of_reference,
    )

    train_loader = MLP_Dataloader(train_dataset, batch_size=options.trajectory.hyperparams.batch_size, shuffle=True, drop_last=True)
    valid_loader = MLP_Dataloader(valid_dataset, batch_size=options.trajectory.hyperparams.batch_size, shuffle=True, drop_last=True)
    
    input_size = (2*number_of_vehicles if options.trajectory.relative_frame_of_reference else 4)*number_of_vehicles + \
            3*number_of_obstacles

    model = MLPModel(
        input_size=input_size,
        output_size=options.trajectory.horizon * number_of_vehicles,
        bound=torch.tensor([math.pi]),
    )

    if PRE_TRAINED is not None:
        model.load_state_dict(torch.load(PRE_TRAINED))

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

    criterion = Loss(options=options)

    return train_data, valid_data, train_loader, valid_loader, model, optimizer, scheduler, criterion

def train_model(options: Options, criterion, train_loader, valid_loader, model, optimizer, scheduler):
    model_path = os.path.join(options.base_directory, "trajectory_mlp", options.trajectory.model_name)

    number_of_vehicles = options.vehicles[0]
    number_of_obstacles = options.obstacles[0]

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
            for (inputs, relative_inputs) in valid_loader:
                model_input = relative_inputs if options.trajectory.relative_frame_of_reference else inputs
                y_hat = model(model_input)
                y_hat = y_hat.reshape(-1, number_of_vehicles, options.trajectory.horizon, 1)
                loss = criterion(y_hat, inputs)
                loss_array = np.append(loss_array, loss.item())
            valid_loss = np.mean(loss_array)
            LOGS["validation_loss"].append(valid_loss)
            return valid_loss

    def train():
        model.train()
        loss_array = np.array([])
        for (inputs, relative_inputs) in train_loader:
            optimizer.zero_grad()
            model_input = relative_inputs if options.trajectory.relative_frame_of_reference else inputs
            y_hat = model(model_input)
            y_hat = y_hat.reshape(-1, number_of_vehicles, options.trajectory.horizon, 1)
            loss = criterion(y_hat, inputs)
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


if __name__ == '__main__':
    options = Options(
        vehicles=[1],
        obstacles=[2],
        base_directory='./output/',
        trajectory=SettingsTrajectory(
        model_name="name",
            horizon=1,
            data_length=30000,
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
                obstacle_cost=5.0,
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
            show_plot=True,
            save_plot=True,
            duration=50
        ),
    )

    # depending on where the file is executed
    options.base_directory = options.base_directory if os.getcwd().endswith('trajectory_prediction') else os.path.join('../', options.base_directory)

    # train the trajectory model
    model, valid_data = train_trajectory_model(options)

    number_of_vehicles = options.vehicles[0]
    number_of_obstacles = options.obstacles[0]
    inference_data = get_data_with_obstacles_one_vehicle(100, number_of_obstacles, number_of_vehicles)

    # load the vehicle model
    vehicle_model_name = "best_model"
    vehicle_model_path = os.path.join(options.base_directory, "vehicle", vehicle_model_name, "model.pth")
    input_size = 4 if options.vehicle.relative_frame_of_reference else 8
    output_size = 2 * options.vehicle.horizon
    vehicle_model = MLPModel(
        input_size=input_size,
        output_size=output_size,
        bound=torch.tensor([1, 0.8]),
    )
    vehicle_model.load_state_dict(torch.load(vehicle_model_path))

    # visualization
    for i in range(100):
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
            is_gnn=False,
            plot_name=str(i),
        )