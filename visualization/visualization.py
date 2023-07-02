import os

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from options import Options


class Visualization:
    def __init__(self, options: Options, number_of_vehicles, number_of_obstacles, is_gnn=False):
        
        self.options = options
        self.cmap = [(0,0,0), (0.5,0,0), (0,0.5,0), (0,0,0.5),
                     (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5), (0.5, 0.5, 0.5)]
        self.number_of_vehicles = number_of_vehicles
        self.number_of_obstacles = number_of_obstacles
        self.is_gnn = is_gnn
        self.plot_attention = self.is_gnn and self.number_of_obstacles > 1
    
    def base_plot(self, starts, targets, obstacles):
        figure_size = self.options.visualization.figure_size
        figure_limit = self.options.visualization.figure_limit
        ticks_step = self.options.visualization.ticks_step
        
        if self.plot_attention:
            self.fig = plt.figure(figsize=(2*figure_size, figure_size))
            self.ax = self.fig.add_subplot(1,2,1)
            self.ax_ = self.fig.add_subplot(1,2,2)
            self.ax_.get_yaxis().set_visible(False)
            
        else:
            self.fig = plt.figure(figsize=(figure_size, figure_size))
            self.ax = self.fig.add_subplot()
            
        self.ax.set_xlim(-figure_limit, figure_limit)
        self.ax.set_ylim([-figure_limit, figure_limit])
        self.ax.set_xticks(np.arange(-figure_limit, figure_limit, step = ticks_step))
        
        self.ax.set_yticks(np.arange(-figure_limit, figure_limit, step = ticks_step))

        self.patch_vehicles = []
        self.patch_vehicles_arrow = []
        self.patch_target = []
        self.patch_target_arrow = []
        self.predicts_opt = []
        self.predicts_model = []

        self.patch_current_target = []

        self.attention_bars = []


        patch_obs = []
        
        start_new = self.car_patch_pos(starts.numpy())
        target_new = self.car_patch_pos(targets.numpy())
        car_width = self.options.visualization.car_size.width
        car_length = self.options.visualization.car_size.length

        for i in range(self.number_of_vehicles):
            # cars
            
            patch_car = mpatches.Rectangle([0,0], car_width, car_length, color=self.cmap[i])
            patch_car.set_xy(start_new[i,:2])
            patch_car.angle = np.rad2deg(start_new[i,2])-90
            self.patch_vehicles.append(patch_car)

            patch_current_target = mpatches.Rectangle(np.array([0,0]), car_width, car_length, 
                                                      color="red", ls='dashed', fill=False)
            patch_current_target.set_xy(np.array([0, 0]))
            patch_current_target.angle = np.rad2deg(0)-90
            self.patch_current_target.append(patch_current_target)
            
            patch_car_arrow = mpatches.FancyArrow(starts[i,0]-0.9*np.cos(starts[i,2]), 
                                                  starts[i,1]-0.9*np.sin(starts[i,2]), 
                                                  1.5*np.cos(starts[i,2]), 
                                                  1.5*np.sin(starts[i,2]), 
                                                  width=0.1, color='w')
            self.patch_vehicles_arrow.append(patch_car_arrow)

            patch_goal = mpatches.Rectangle([0,0], car_width, car_length, color=self.cmap[i], 
                                            ls='dashdot', fill=False)
            
            patch_goal.set_xy(target_new[i,:2])
            patch_goal.angle = np.rad2deg(target_new[i,2])-90
            self.patch_target.append(patch_goal)
            
            patch_goal_arrow = mpatches.FancyArrow(targets[i,0]-0.9*np.cos(targets[i,2]), 
                                                   targets[i,1]-0.9*np.sin(targets[i,2]), 
                                                   1.5*np.cos(targets[i,2]), 
                                                   1.5*np.sin(targets[i,2]), 
                                                   width=0.1, 
                                                   color=self.cmap[i])
            self.patch_target_arrow.append(patch_goal_arrow)
            
            self.next_target_points = self.ax.scatter([], [])


            self.ax.add_patch(patch_car)
            self.ax.add_patch(patch_goal)
            self.ax.add_patch(patch_car_arrow)
            self.ax.add_patch(patch_goal_arrow)
            self.ax.add_patch(patch_current_target)

            self.frame = plt.text(12, 12, "", fontsize=15)
            
            if i == 0:
                predict_model, = self.ax.plot([], [], 'b--', linewidth=1, label="Model Prediction")
            else:
                predict_model, = self.ax.plot([], [], 'b--', linewidth=1, label="_Model Prediction")
            self.predicts_model.append(predict_model)
    
            vehicle_mark, = self.ax.plot([], [], color=self.cmap[i], marker='.', linewidth=1, label=f"color of vehicle {i+1}")

            if self.plot_attention:
                self.attention_bars.append(self.ax_.bar('Attention Vehicle {}'.format(i+1), [1]*len(obstacles)))
        
        bottom = 0
        for i, obs in enumerate(obstacles):
            patch_obs.append(mpatches.Circle(obs[:2], obs[2], color=self.cmap[i], fill=True))
            self.ax.add_patch(patch_obs[-1])
            obstacle_mark, = self.ax.plot([], [], color=self.cmap[i], marker='.', linewidth=1, label=f"color of obstacle {i+1}")

        self.ax.legend(loc='upper left', fontsize=12)
    
    
    def create_video(self, starts, targets, obstacles, data, predict_model, current_target=None, trajectory_positions=None, data_attention_gnn=None, plot_name=None):
        self.base_plot(starts, targets, obstacles)
        self.data = data
        self.current_target_data = current_target
        self.trajectory_positions = trajectory_positions
        self.attention = data_attention_gnn
        self.predict_model_data = predict_model

        car_animation = animation.FuncAnimation(self.fig, self.update_plot, frames=range(len(data)-1), interval=100, repeat=True, blit=False)
        base_folder = os.path.join(self.options.base_directory, "trajectory_gnn" if self.is_gnn else "vehicle")
        if self.options.visualization.save_plot:
            model_dir = os.path.join(base_folder, \
                                     self.options.trajectory.model_name if self.options.trajectory is not None \
                                          else self.options.vehicle.model_name)
            
            car_animation.save(os.path.join(model_dir, plot_name + ".gif"))
            
        if self.options.visualization.show_plot:
            plt.show()

    def update_plot(self, num):
        data = self.data[num,...]
        model_prediction = self.predict_model_data[num, :, :, :2]  
        current_target = self.current_target_data[num,...]
        current_trajectory_positions = self.trajectory_positions[num, :, :]
        self.current_targets = [None] * self.number_of_vehicles
        for i in range(self.number_of_vehicles):
            data_current_target = self.car_patch_pos(current_target[i, :])
            self.patch_current_target[i].set_xy(data_current_target[:2])
            self.patch_current_target[i].angle = np.rad2deg(data_current_target[2])-90

            self.next_target_points.set_offsets(current_trajectory_positions.reshape(-1, 2))
            data_ = self.car_patch_pos(data[i][None,...])
            self.patch_vehicles[i].set_xy(data_[0,:2])
            self.patch_vehicles[i].angle = np.rad2deg(data_[0,2])-90
            self.patch_vehicles_arrow[i].set_data(x=data[i,0]-0.9*np.cos(data[i,2]), 
                                                    y=data[i,1]-0.9*np.sin(data[i,2]), 
                                                    dx=1.5*np.cos(data[i,2]), 
                                                    dy=1.5*np.sin(data[i,2]))

            self.predicts_model[i].set_data(model_prediction[:, i, 0], model_prediction[:, i, 1])

            if self.plot_attention:
                data_bar = np.array([value[num] for (key, value) in self.attention.items()]).reshape(self.number_of_vehicles, self.number_of_obstacles)
                bottom = 0

                self.ax_.set_ylim(0, np.max(np.sum(data_bar, axis=1))*1.1)
                for index, rect in enumerate(self.attention_bars[i]):
                    rect.set_height(data_bar[i][index])
                    rect.set_y(bottom)
                    rect.set_color(self.cmap[index])
                    bottom += data_bar[i][index]
    
    def car_patch_pos(self, posture):
        
        posture_new = posture.copy()
        
        posture_new[...,0] = posture[...,0] - np.sin(posture[...,2])*(self.options.visualization.car_size.width/2) \
                                        - np.cos(posture[...,2])*(self.options.visualization.car_size.length/2)
        posture_new[...,1] = posture[...,1] + np.cos(posture[...,2])*(self.options.visualization.car_size.width/2) \
                                        - np.sin(posture[...,2])*(self.options.visualization.car_size.length/2)
        
        return posture_new
    