import sys 
import os
import torch
import pprint
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options import Options, SettingsVehicle, LossVehicle, Hyperparams, SettingsVisualization, Size
from data import get_close_data, get_inference_data
from vehicle.dataset import MLP_Dataset_Vehicle, MLP_Dataloader_Vehicle, change_to_relative_frame_of_reference_vehicle
from mlp import MLPModel
from vehicle.loss import Loss
from visualization.simulation import sim_run

TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.2

VEHICLE_SETTINGS = SettingsVehicle(
        horizon=5,
        data_length=10000,
        normal_percentage=0.5,
        parking_percentage=0.5,
        relative_frame_of_reference=True,
        model_name="base_model",
        hyperparams=Hyperparams(
            epochs=1000,
            patience=70,
            batch_size=512,
            learning_rate=1e-4,
            learning_rate_patience=40,
            learning_rate_factor=0.2,
            learning_rate_minimum=1e-10,
            weight_decay=1e-7,
            cool_down=50,
        ),
        loss=LossVehicle(
            distance_cost=1.0,
            angle_cost=1.0,
            just_last_step=False,
            triangluar_loss=False,
            velocity_cost=0.1,
        ),
)

def train_vehicle_model(options: Options):
    model_path = os.path.join(options.base_directory, "vehicle", options.vehicle.model_name)
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
    number_of_vehicles = options.vehicles[0]
    number_of_obstacles = options.obstacles[0]
    train_data = get_close_data(int(options.vehicle.data_length*TRAIN_SPLIT)+1)
    valid_data = get_close_data(int(options.vehicle.data_length*VALID_SPLIT)+1)

    train_dataset = MLP_Dataset_Vehicle(
        data=train_data,
        relative_frame_of_reference=options.vehicle.relative_frame_of_reference,
        number_of_vehicle=number_of_vehicles,
        number_of_obstacles=number_of_obstacles,
    )
    
    valid_dataset = MLP_Dataset_Vehicle(
        data=train_data,
        relative_frame_of_reference=options.vehicle.relative_frame_of_reference,
        number_of_vehicle=number_of_vehicles,
        number_of_obstacles=number_of_obstacles,
    )

    train_loader = MLP_Dataloader_Vehicle(
        train_dataset,
        batch_size=options.vehicle.hyperparams.batch_size,
        shuffle=True,
        drop_last=True,
        relative_frame_of_reference=options.vehicle.relative_frame_of_reference,
    )

    valid_loader = MLP_Dataloader_Vehicle(
        valid_dataset,
        batch_size=options.vehicle.hyperparams.batch_size,
        shuffle=True,
        drop_last=True,
        relative_frame_of_reference=options.vehicle.relative_frame_of_reference,
    )   

    input_size = 4 if options.vehicle.relative_frame_of_reference else 8
    input_size += number_of_obstacles * 3
    output_size = 2 * number_of_vehicles * options.vehicle.horizon

    model = MLPModel(
        input_size=input_size,
        output_size=output_size,
        bound=torch.tensor([1, 0.8]),
    )

    optimizer = Adam(
        model.parameters(), 
        lr = options.vehicle.hyperparams.learning_rate, 
        weight_decay = options.vehicle.hyperparams.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer = optimizer, mode = 'min', verbose = True, 
        patience = options.vehicle.hyperparams.learning_rate_patience,
        factor = options.vehicle.hyperparams.learning_rate_factor, 
        min_lr = options.vehicle.hyperparams.learning_rate_minimum,
        cooldown=10
    )
    criterion = Loss(options=options)

    return train_data, valid_data, train_loader, valid_loader, model, optimizer, scheduler, criterion

def train_model(options: Options, criterion, train_loader, valid_loader, model, optimizer, scheduler):
    model_path = os.path.join(options.base_directory, "vehicle", options.vehicle.model_name)
    plt.clf()
    
    with open(os.path.join(model_path, "options.txt"), 'w') as f:
        pprint.pprint(vars(options.vehicle), stream=f)

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
                model_input = relative_inputs if options.vehicle.relative_frame_of_reference else inputs
                y_hat = model(model_input)
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
            model_input = relative_inputs if options.vehicle.relative_frame_of_reference else inputs
            y_hat = model(model_input)
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
    patience_f = options.vehicle.hyperparams.patience

    for epoch in tqdm(range(options.vehicle.hyperparams.epochs)):
        train_loss = train()
        valid_loss = validate()

        if epoch > options.vehicle.hyperparams.cool_down:
            scheduler.step(train_loss)
            
        msg = f'Epoch: {epoch+1}/{options.vehicle.hyperparams.epochs} | Train Loss: {train_loss:.6} | Valid Loss: {valid_loss:.6}'
        print(msg)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_f = options.vehicle.hyperparams.patience
            torch.save(model.state_dict(), os.path.join(model_path, "model.pth"))
            print('Model Saved!')
                
        elif epoch > options.vehicle.hyperparams.cool_down:
            patience_f -= 1
            if patience_f == 0:
                print(f'Early stopping (no improvement since {options.vehicle.hyperparams.patience} models) | Best Valid Loss: {best_loss:.6f}')
                break

    plt.plot(LOGS["training_loss"], label='Training Loss')
    plt.plot(LOGS["validation_loss"], label='Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(model_path, "learning.png"))
    plt.show()
    return model

if __name__ == '__main__':
    options = Options(
        vehicles=[1],
        obstacles=[0],
        base_directory='./output/',
        trajectory=None,
        vehicle=VEHICLE_SETTINGS,
        visualization=SettingsVisualization(
            figure_size=8,
            figure_limit=35,
            ticks_step=5,
            car_size=Size(width=1.0, length=2.5),
            show_plot=True,
            save_plot=True,
            duration=100
        ),
    )
    
    # depending on where the file is executed
    options.base_directory = options.base_directory if os.getcwd().endswith('trajectory_prediction') else os.path.join('../', options.base_directory)
    
    # train the vehicle model
    model, valid_data = train_vehicle_model(options)

    # get data for inference
    inference_data_vehicles, inference_data_obstacles = get_inference_data(100, options.vehicles[0], 1)

    # visualization
    for i in range(100):
        sim_run(
            inference_data_vehicles[i, :, :4].numpy(),
            inference_data_vehicles[i, :, 4:7].numpy(),
            inference_data_obstacles[i,:,:3].numpy(),
            options,
            model=model,
            trajectory_model=None,
            relative_frame_of_reference_converter_vehicle=change_to_relative_frame_of_reference_vehicle,
            relative_frame_of_reference_converter_trajectory=None,
            is_gnn=False,
            plot_name=str(i),
        )


    