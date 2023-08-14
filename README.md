# Vehicle Control

This project aims to train a model that is capable of guiding a vehicle from a starting position to a target position while avoiding collisions.

## Dynamic Vehicle Equation

The dynamic vehicle control model simulates the vehicles based on control inputs. Position, angle, and velocity of the vehicle depend on the previous state. The model should predict the correct steering and pedal inputs to guide vehicles to their target position. The challenge here is the fact that such a vehicle model is a non-holonomic system, meaning that the position can only indirectly be influenced.

x_{t+1} = x_t + v_t * cos(psi_t) * dt

y_{t+1} = y_t + v_t * sin(psi_t) * dt

psi_{t+1} = psi_t + v_t * tan(steering) * 0.5 * dt

v_{t+1} = 0.99 * v_t + pedal * dt

## Vehicle Model

The vehicle model takes an intermediate target as input and predicts the steering and pedal input which will subsequently be the input for the dynamic vehicle equation computing the next state. The vehicle model is an MLP that is applied to each vehicle separately. Each vehicle is encoded by 4 values describing its state (x,y,angle,velocity) and 4 values describing its target state (x,y,angle,velocity). The vehicle model predicts for each horizon step the steering and pedal input

### Loss

The vehicle model is trained in an unsupervised manner. The loss is based on the difference of the position, angle, and velocity of the current state and the target state.


## Trajectory Model

The trajectory model takes the set of vehicles and obstacles as input and predicts the next intermediate target for each vehicle. The model can either be an MLP with a fixed number of vehicles and obstacles or a GNN with an arbitrary number of them. The GNN connects all vehicles and obstacles in one simulation and uses the [Transformer Conv](https://arxiv.org/abs/2009.03509) edge update to model their relations. The plots show the attention that each vehicle spends towards the obstacles.

### Loss

The trajectory model is trained in an unsupervised manner. The loss is based on the current position of the car to the global target as well as the distance to each obstacle and other vehicles.


## Results

![One vehicle being guided to its target position while avoiding an obstacle](./output/example_plots/0.gif)

![One vehicle being guided to its target position while avoiding four obstacle](./output/example_plots/1.gif)

![Two vehicles being guided to their target positions while avoiding one obstacle](./output/example_plots/20.gif)

## Acknowledgement
I would like to thank [Yining Ma](https://github.com/yininghase), as we stared this project together.
