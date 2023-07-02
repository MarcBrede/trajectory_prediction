import numpy as np
import torch
import itertools

from visualization.visualization import Visualization
from visualization.trajectory import get_next_target, get_angle, get_velocity
from options import Options


def sim_run(
        starts,
        targets,
        obstacles,
        options: Options, 
        model = None,
        trajectory_model = None,
        relative_frame_of_reference_converter_vehicle = None,
        relative_frame_of_reference_converter_trajectory = None,
        is_gnn=False,
        plot_name=None
    ):
    
    state_i = np.array([starts])
    number_of_vehicles = len(starts)
    number_of_obstacles = len(obstacles)
    trajectory_horizon = options.trajectory.horizon if trajectory_model is not None else 1
    vehicle_horizon = options.vehicle.horizon
    
    data_next_target_points = np.empty((0, number_of_vehicles, trajectory_horizon, 2))
    data_next_target = np.empty((0, number_of_vehicles, 4))
    data_predict_model_info = np.empty((0, vehicle_horizon, number_of_vehicles, 4))
    data_attention_gnn = {(key, value): [] for (key, value) in itertools.product(range(1, number_of_vehicles+1), range(1, number_of_obstacles+1))}

    for i in range(1, options.visualization.duration+1):
        starts = torch.tensor(state_i[-1], dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        obstacles = torch.tensor(obstacles, dtype=torch.float32)
        current_targets = np.empty((1, number_of_vehicles, 4))

        # get intermediate target
        if trajectory_model is not None:
            trajectory_model.eval()
            if relative_frame_of_reference_converter_trajectory is not None:
                vehicles = torch.concatenate((starts[:, :2], targets[:, :2]), axis=1)[None, : ,:]
                # if 
                if len(obstacles) == 0:
                    model_input = relative_frame_of_reference_converter_trajectory(vehicles, torch.zeros((1, 0, 3)))
                else:
                    model_input = relative_frame_of_reference_converter_trajectory(vehicles, obstacles[None, :, :])
                vehicles = model_input[:, :number_of_vehicles, :2*number_of_vehicles].reshape(-1, 2*number_of_vehicles + (2*(number_of_vehicles-1))*number_of_vehicles)
                if len(obstacles) == 0:
                    model_input = vehicles
                else:
                    obstacles = model_input[:, number_of_vehicles:, :3].reshape(-1, 3*number_of_vehicles)
                    model_input = torch.cat((vehicles, obstacles), dim=1)
            elif not is_gnn:
                vehicles = torch.concatenate((starts[:, :2], targets[:, :2]), axis=-1)
                vehicles = vehicles.reshape(-1, number_of_vehicles*4)
                if number_of_obstacles > 0:
                    obstacles = obstacles.reshape(-1, 3*number_of_obstacles)
                    model_input = torch.cat((vehicles, obstacles), dim=1)
                else:
                    model_input = vehicles

            if is_gnn:
                data = torch.zeros((number_of_vehicles + number_of_obstacles, 5))
                data[:number_of_vehicles, [1, 2]] = starts[:,:2]
                data[:number_of_vehicles, [3, 4]] = targets[:,:2]
                data[number_of_vehicles:, [1, 2, 3]] = obstacles
                data[number_of_vehicles:, 0] = 1
                batch = torch.tensor([[number_of_vehicles + number_of_obstacles, number_of_vehicles]])
                # edge_template = generate_edge_template(mpc.num_vehicle, mpc.num_obstacle)
                # edges_vehicles, edges_obstacles = get_edges(batch, edge_template)
                all_predictions, prediction, _, edges_vehicle, edges_obstacle, attention = trajectory_model(data, batch)
                if options.trajectory.classes is not None:
                    prediction = torch.argmax(prediction.reshape(-1, trajectory_horizon, options.trajecotry.classes)) * (2*np.pi/options.trajecotry.classes)
                
                # let index start at 1 
                attention_permutations = [np.array([i, j]) for i in range(1, number_of_vehicles+1) for j in range(1, number_of_obstacles+1)]
                for i, key in enumerate(attention_permutations):
                    searched_edge = np.array([key[1], key[0]]) + [number_of_vehicles-1, -1]
                    row_index = torch.where(torch.all(attention[0].T == torch.tensor(searched_edge), dim=-1))[0]
                    data_attention_gnn[tuple(key)].append(attention[1][row_index].item())
                # attention_list = [(vehicle_index.item(), obstacle_index.item()-mpc.num_vehicle) for (obstacle_index, vehicle_index) in attention[0].T + 1]
                # for i, key in enumerate(attention_list):
                #     data_attention_gnn[key].append(attention[1][i].item())
            else:
                prediction = trajectory_model(model_input)

            TRAJECTORY_PIONTS_LENGTH = 2
            prediction = prediction.reshape(number_of_vehicles, trajectory_horizon, 1)
            prediction = torch.cat((torch.cos(prediction), torch.sin(prediction)), dim=-1)
            trajectory_positions = torch.zeros((number_of_vehicles, trajectory_horizon+1, 2))
            trajectory_positions[:, 0, :] = starts[:, :2]
            for j in range(trajectory_horizon):
                trajectory_positions[:, j+1, :] = trajectory_positions[:, j, :] + (prediction[:, j, :] * TRAJECTORY_PIONTS_LENGTH)
            trajectory_positions = trajectory_positions[:, 1:, :]
            next_target_position = trajectory_positions[:, 0, :]

            next_angle = get_angle(starts[:, :2], next_target_position)
            next_velocity = get_velocity(starts[:, :], targets[:, :2])
            next_target = torch.cat((next_target_position, next_angle, next_velocity), axis=1)

            # replace close to acutal targets with actual targets
            ACTUAL_TARGET_THRESHOLD = 4
            mask = torch.linalg.norm(next_target_position - targets[:, :2], dim=-1) < ACTUAL_TARGET_THRESHOLD
            next_target[mask, :2] = targets[mask, :2]

            data_next_target_points = np.concatenate((data_next_target_points, trajectory_positions.detach().cpu().numpy()[None, :,:]), axis=0)
            data_next_target = np.concatenate((data_next_target, next_target.detach().cpu().numpy()[None,:,:]), axis=0)
        else:
            next_target = get_next_target(
                position=starts,
                target=targets,
                obstacle=obstacles,
                vehicle_obstacles=None,
            )

            data_next_target_points = np.concatenate((data_next_target_points, next_target[:, :2][None, None, :,:]), axis=0)
            data_next_target = np.concatenate((data_next_target, next_target[None,:,:]), axis=0)
        
        # get vehicel control
        model.eval()
        if relative_frame_of_reference_converter_vehicle is not None:
            model_input = torch.cat((starts, next_target), dim=1)
            model_input = relative_frame_of_reference_converter_vehicle(
                model_input, 
                num_of_vehicles=number_of_vehicles,
                num_of_obstacles=0,
            )
            u_model = torch.zeros((vehicle_horizon, number_of_vehicles, 2))
            for i in range(len(model_input)):
                u_model[:, i, :] = model(model_input[i][None, :]).reshape(vehicle_horizon, 2)
        else:
            u_model = torch.zeros((vehicle_horizon, number_of_vehicles, 2))
            model_input = torch.cat((starts, next_target), dim=1)
            for i in range(len(model_input)):
                u_model[:, i, :] = model(model_input[i][None, :]).reshape(vehicle_horizon, 2)
            
        u_model = u_model.reshape(1, vehicle_horizon, number_of_vehicles, 2)
        position = torch.tensor(state_i[-1], dtype=torch.float32)[None, :, :]
        prediction = position[None, ...]
        for i in range(vehicle_horizon):
            position = one_step_forward(position, u_model[:, i, :, :])
            prediction = torch.cat((prediction, position[None, ...]))
        
        current_position = prediction[1, :, :, :4].detach().numpy()
        prediction = prediction[1:, ...]
        prediction = prediction.transpose(0, 1)

        data_predict_model_info = np.concatenate((data_predict_model_info, prediction.detach().numpy()))
        
        ### Simulation ###
        state_i = np.concatenate((state_i, current_position), axis=0)
    
    
    if options.visualization.save_plot or options.visualization.show_plot:
        visualization = Visualization(options, number_of_vehicles, number_of_obstacles, is_gnn=is_gnn)
        visualization.create_video(
            starts,
            targets,
            obstacles,
            state_i, 
            data_predict_model_info,
            current_target=data_next_target,
            trajectory_positions=data_next_target_points,
            data_attention_gnn=data_attention_gnn,
            plot_name=plot_name
        )

DT = 0.2
def one_step_forward(position, controls):
    x_t = position[..., 0] + position[..., 3] * torch.cos(position[..., 2]) * DT
    y_t = position[..., 1] + position[..., 3] * torch.sin(position[..., 2]) * DT
    psi_t = (
        position[..., 2] + position[..., 3] * DT * torch.tan(controls[..., 1]) / 2.0
    )
    psi_t = (psi_t + np.pi) % (2 * np.pi) - np.pi
    v_t = 0.99 * position[..., 3] + controls[..., 0] * DT

    return torch.cat(
        (x_t[..., None], y_t[..., None], psi_t[..., None], v_t[..., None]), dim=-1
    )

