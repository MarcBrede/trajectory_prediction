import torch
import numpy as np
from options import Options
from torch import nn


DT = 0.2
class Loss(nn.Module):
    def __init__(self, options: Options) -> None:
        super().__init__()
        self.options = options

    def forward(self, y_hat: torch.tensor, inputs: torch.tensor) -> float:
        number_of_vehicles = self.options.vehicles[0]
        number_of_obstacles = self.options.obstacles[0]

        y_hat = y_hat.reshape(
            -1, self.options.vehicle.horizon, number_of_vehicles, 2
        )
        target = inputs[:, 4:7][:, None, :]
        position = inputs[:, :4][:, None, :]
        obstacle = inputs[:, 7:]
        prediction = position[None, ...]
        for i in range(self.options.vehicle.horizon):
            position = self.one_step_forward(position, y_hat[:, i, :, :])
            prediction = torch.cat((prediction, position[None, ...]))

        current_position = prediction[0, :, :, :2]
        prediction = prediction[1:, ...]
        prediction = prediction.transpose(0, 1)

        loss = 0

        # distance
        if not self.options.vehicle.loss.just_last_step:
            loss += (
                torch.mean(
                    torch.linalg.norm(
                        prediction[:, :, :, :2] - target[:, None, :, :2], dim=-1, ord=2
                    )
                )
                * self.options.vehicle.loss.distance_cost
            )
        else:
            loss += (
                torch.mean(
                    torch.linalg.norm(
                        prediction[:, -1, :, :2] - target[:, None, :, :2], dim=-1, ord=2
                    )
                )
                * self.options.vehicle.loss.distance_cost
            )

        # angle
        if not self.options.vehicle.loss.just_last_step:
            angle_diff_1 = (prediction[:, :, :, 2] - target[:, None, :, 2])[..., None] % (
                2 * np.pi
            )
        else:
            angle_diff_1 = (prediction[:, -1, :, 2] - target[:, None, :, 2])[..., None] % (
                2 * np.pi
            )
        angle_diff_2 = 2 * np.pi - angle_diff_1
        angle_diff = (
            torch.amin(torch.concat((angle_diff_1, angle_diff_2), dim=-1), dim=-1) ** 2
        )
        loss += torch.mean(angle_diff) * self.options.vehicle.loss.angle_cost

        # obstacle
        if number_of_obstacles > 0 and (not self.options.vehicle.loss.target_velocity):
            if not self.options.vehicle.loss.triangluar_loss:
                obstacle_distance = torch.linalg.norm(
                    prediction[:, :, :, :2] - obstacle[:, None, None, :2], ord=2, dim=-1
                )
                # A
                # loss += (1/obstacle_distance).mean() * self.options.loss.obstacle_cost
                # B
                # obstacles_distance_near = obstacle_distance[obstacle_distance < (obstacle[:, None, 2][:, None, :] + self.options.loss.obstacle_radius)]
                # C
                obstacles_distance_near = obstacle_distance[obstacle_distance < obstacle[:, None, 2][:, None, :]]
                if len(obstacles_distance_near) > 0:
                    loss += torch.mean(1 / obstacles_distance_near) * self.options.vehicle.loss.obstacle_cost
            else:
                loss += self.triangular_loss(prediction, current_position, obstacle)
        
        # velocity 
        loss += torch.mean(torch.abs(prediction[:, -1, :, 3] - inputs[:, -1][:, None])) * self.options.vehicle.loss.velocity_cost

        return loss

    def one_step_forward(self, position, controls):
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
    
    def point_in_triangle(self, p, A1, A2, A3):
        v0 = A3 - A1
        v1 = A2 - A1
        v2 = p - A1

        dot00 = torch.sum(v0 * v0, dim=-1)
        dot01 = torch.sum(v0 * v1, dim=-1)
        dot02 = torch.sum(v0 * v2, dim=-1)
        dot11 = torch.sum(v1 * v1, dim=-1)
        dot12 = torch.sum(v1 * v2, dim=-1)

        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        return ((u >= 0) & (v >= 0) & (u + v < 1))

    def triangular_loss(self, prediction, current_position, obstacle):
        ADDITIONAL_OFFSET = 0
        loss = 0
        # Triangle defined by three lines
        # A
        A1 = torch.stack((obstacle[:, 0], (obstacle[:, 1] + obstacle[:, 2] + ADDITIONAL_OFFSET)), dim=-1)[:, None, :]
        A2 = torch.stack((obstacle[:, 0], (obstacle[:, 1] - obstacle[:, 2] - ADDITIONAL_OFFSET)), dim=-1)[:, None, :]
        A3 = current_position 
        # # B 
        B1 = torch.stack(((obstacle[:, 0] + obstacle[:, 2] + ADDITIONAL_OFFSET), obstacle[:, 1]), dim=-1)[:, None, :]
        B2 = torch.stack(((obstacle[:, 0] - obstacle[:, 2] - ADDITIONAL_OFFSET), obstacle[:, 1]), dim=-1)[:, None, :]
        B3 = current_position 

        # from PIL import Image
        # ps = torch.cat((torch.arange(start=-20, end=20, step=0.01).repeat(4000, 1)[:, :, None], torch.arange(start=-20, end=20, step=0.01).repeat(4000, 1).T[:,:,None]), dim=-1)
        # ps = ps.reshape(4000*4000, 2)[None, :, None, :]
        # ps = self.point_in_triangle(ps, A1[0, None, :, :], A2[0, None, :, :], A3[0, None, :, :])
        # ps = ps.reshape(4000, 4000)
        # ps = ps.numpy()
        # ps = np.array(ps, dtype=np.uint8) * 255    
        # img = Image.fromarray(ps, 'L')
        # img.show('my.png')

        A_mask = self.point_in_triangle(prediction[:, :, :, :2], A1[:, None, :, :], A2[:, None, :, :], A3[:, None, :, :])
        B_mask = self.point_in_triangle(prediction[:, :, :, :2], B1[:, None, :, :], B2[:, None, :, :], B3[:, None, :, :])

        # # get mid point of triangle
        mid_point_A = (A1 + A2 + A3) / 3
        mid_point_B = (B1 + B2 + B3) / 3

        # Formulat according to: https://wikimedia.org/api/rest_v1/media/math/render/svg/aad3f60fa75c4e1dcbe3c1d3a3792803b6e78bf6
        if torch.sum(A_mask) > 0:
            # distance to A1 A2
            x_1 = A1[:, :, 0].squeeze()
            y_1 = A1[:, :, 1].squeeze()
            x_2 = A2[:, :, 0].squeeze()
            y_2 = A2[:, :, 1].squeeze()
            x_0 = prediction[:, :, :, 0]
            y_0 = prediction[:, :, :, 1]
            tmp = torch.mul((x_2 - x_1)[:, None, None], (y_1[:, None, None] - y_0)) - torch.mul((x_1[:, None, None] - x_0), (y_2 - y_1)[:, None, None])
            distance_1 = torch.abs(tmp) / torch.linalg.norm(A2 - A1, dim=-1)[:, None, :]

            # distance to A1 A3
            x_1 = A1[:, :, 0].squeeze()
            y_1 = A1[:, :, 1].squeeze()
            x_2 = A3[:, :, 0].squeeze()
            y_2 = A3[:, :, 1].squeeze()
            tmp = torch.mul((x_2 - x_1)[:, None, None], (y_1[:, None, None] - y_0)) - torch.mul((x_1[:, None, None] - x_0), (y_2 - y_1)[:, None, None])
            distance_2 = torch.abs(tmp) / torch.linalg.norm(A3 - A1, dim=-1)[:, None, :]

            # distance to A2 A3
            x_1 = A2[:, :, 0].squeeze()
            y_1 = A2[:, :, 1].squeeze()
            x_2 = A3[:, :, 0].squeeze()
            y_2 = A3[:, :, 1].squeeze()
            tmp = torch.mul((x_2 - x_1)[:, None, None], (y_1[:, None, None] - y_0)) - torch.mul((x_1[:, None, None] - x_0), (y_2 - y_1)[:, None, None])
            distance_3 = torch.abs(tmp) / torch.linalg.norm(A3 - A2, dim=-1)[:, None, :]

            shortest_distance_to_triangle = torch.min(torch.cat((distance_1, distance_2, distance_3), dim=-1), dim=-1)[0]

            tmp = shortest_distance_to_triangle[A_mask.squeeze()] / torch.linalg.norm(prediction[..., :2] - mid_point_A[:, None, : ,:], dim=-1)[A_mask]
            loss += torch.mean(tmp) * self.options.loss.obstacle_cost

        if torch.sum(B_mask) > 0:
            # same as above with B
            x_1 = B1[:, :, 0].squeeze()
            y_1 = B1[:, :, 1].squeeze()
            x_2 = B2[:, :, 0].squeeze()
            y_2 = B2[:, :, 1].squeeze()
            x_0 = prediction[:, :, :, 0]
            y_0 = prediction[:, :, :, 1]
            tmp = torch.mul((x_2 - x_1)[:, None, None], (y_1[:, None, None] - y_0)) - torch.mul((x_1[:, None, None] - x_0), (y_2 - y_1)[:, None, None])
            distance_1 = torch.abs(tmp) / torch.linalg.norm(B2 - B1, dim=-1)[:, None, :]

            x_1 = B1[:, :, 0].squeeze()
            y_1 = B1[:, :, 1].squeeze()
            x_2 = B3[:, :, 0].squeeze()
            y_2 = B3[:, :, 1].squeeze()
            tmp = torch.mul((x_2 - x_1)[:, None, None], (y_1[:, None, None] - y_0)) - torch.mul((x_1[:, None, None] - x_0), (y_2 - y_1)[:, None, None])
            distance_2 = torch.abs(tmp) / torch.linalg.norm(B3 - B1, dim=-1)[:, None, :]

            x_1 = B2[:, :, 0].squeeze()
            y_1 = B2[:, :, 1].squeeze()
            x_2 = B3[:, :, 0].squeeze()
            y_2 = B3[:, :, 1].squeeze()
            tmp = torch.mul((x_2 - x_1)[:, None, None], (y_1[:, None, None] - y_0)) - torch.mul((x_1[:, None, None] - x_0), (y_2 - y_1)[:, None, None])
            distance_3 = torch.abs(tmp) / torch.linalg.norm(B3 - B2, dim=-1)[:, None, :]
            
            shortest_distance_to_triangle = torch.min(torch.cat((distance_1, distance_2, distance_3), dim=-1), dim=-1)[0]

            tmp = shortest_distance_to_triangle[B_mask.squeeze()] / torch.linalg.norm(prediction[..., :2] - mid_point_B[:, None, : ,:], dim=-1)[B_mask]
            loss += torch.mean(tmp) * self.options.loss.obstacle_cost
        return loss

    def points_in_triangle(self, position, positions, obstacles):
        ADDITIONAL_OFFSET = 1
        A1 = torch.stack((obstacles[:, 0], (obstacles[:, 1] + obstacles[:, 2] + ADDITIONAL_OFFSET)), dim=-1)
        A2 = torch.stack((obstacles[:, 0], (obstacles[:, 1] - obstacles[:, 2] - ADDITIONAL_OFFSET)), dim=-1)
        A3 = position

        B1 = torch.stack(((obstacles[:, 0] + obstacles[:, 2] + ADDITIONAL_OFFSET), obstacles[:, 1]), dim=-1)
        B2 = torch.stack(((obstacles[:, 0] - obstacles[:, 2] - ADDITIONAL_OFFSET), obstacles[:, 1]), dim=-1)
        B3 = position

        A1 = A1[None, None, :, :]
        A2 = A2[None, None, :, :]
        A3 = A3[None, None, :, :]
        B1 = B1[None, None, :, :]
        B2 = B2[None, None, :, :]
        B3 = B3[None, None, :, :]
        positions = positions[:, None, None, :]

        v0 = A3 - A1
        v1 = A2 - A1
        v2 = positions - A1

        dot00 = torch.sum(v0 * v0, dim=-1)
        dot01 = torch.sum(v0 * v1, dim=-1)
        dot02 = torch.sum(v0 * v2, dim=-1)
        dot11 = torch.sum(v1 * v1, dim=-1)
        dot12 = torch.sum(v1 * v2, dim=-1)

        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        A_mask = (u >= 0) & (v >= 0) & (u + v < 1)
        return A_mask.squeeze(2)

    # def triangular_loss_positions(self, positions, obstacles):
    #     ADDITIONAL_OFFSET = 0
    #     A1 = torch.stack((obstacles[:, 0], (obstacles[:, 1] + obstacles[:, 2] + ADDITIONAL_OFFSET)), dim=-1)
    #     A2 = torch.stack((obstacles[:, 0], (obstacles[:, 1] - obstacles[:, 2] - ADDITIONAL_OFFSET)), dim=-1)

    #     B1 = torch.stack(((obstacles[:, 0] + obstacles[:, 2] + ADDITIONAL_OFFSET), obstacles[:, 1]), dim=-1)
    #     B2 = torch.stack(((obstacles[:, 0] - obstacles[:, 2] - ADDITIONAL_OFFSET), obstacles[:, 1]), dim=-1)

    #     A1 = A1[None, :, :]
    #     A2 = A2[None, :, :]
    #     B1 = B1[None, :, :]
    #     B2 = B2[None, :, :]
    #     positions = positions[:, None, :]

    #     mid_points_A = (A1 + A2 + positions) / 2
    #     mid_points_B = (B1 + B2 + positions) / 2

    #     x_1 = A1[:, :, 0][None, :, :, None]
    #     y_1 = A1[:, :, 1][None, :, :, None]
    #     x_2 = A2[:, :, 0][None, :, :, None]
    #     y_2 = A2[:, :, 1][None, :, :, None]
    #     x_0 = positions[:, :, 0][:, None, :, None]
    #     y_0 = positions[:, :, 1][:, None, :, None]
    #     tmp = torch.mul((x_2 - x_1), (y_1 - y_0)) - torch.mul(x_1 - x_0, (y_2 - y_1))
    #     distance_1 = torch.abs(tmp) / torch.linalg.norm(A2 - A1, dim=-1)[None, :, :, None]

    #     x_1 = A1[:, :, 0][None, :, :, None]
    #     y_1 = A1[:, :, 1][None, :, :, None]
    #     x_2 = positions[:, :, 0][:, None, :, None]
    #     y_2 = positions[:, :, 1][:, None, :, None]
    #     tmp = torch.mul((x_2 - x_1), (y_1 - y_0)) - torch.mul(x_1 - x_0, (y_2 - y_1))
    #     distance_2 = torch.abs(tmp) / torch.linalg.norm(positions[:, :, None, :] - A1[None, :, :, :], dim=-1)[..., None]

    #     x_1 = A2[:, :, 0][None, :, :, None]
    #     y_1 = A2[:, :, 1][None, :, :, None]
    #     x_2 = positions[:, :, 0][:, None, :, None]
    #     y_2 = positions[:, :, 1][:, None, :, None]
    #     tmp = torch.mul((x_2 - x_1), (y_1 - y_0)) - torch.mul(x_1 - x_0, (y_2 - y_1))
    #     distance_3 = torch.abs(tmp) / torch.linalg.norm(positions[:, :, None, :] - A2[None, :, :, :], dim=-1)[..., None]

        
    #     print('test')

    # def only_distance_obstacle_loss(self, positions, target, obstacles):
    #     loss = torch.linalg.norm(positions[:, None, :] - target[None, :, :], dim=-1)
    #     loss += self.triangular_loss_positions(positions, obstacles)
