from torch import nn
import torch

HIDDEN_SIZE = 400

class MLPModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, bound: torch.tensor) -> None:
        super().__init__()
        self.block1 = LinearBlock(input_size, HIDDEN_SIZE, activation="relu")
        self.block2 = ResidualBlock(HIDDEN_SIZE, HIDDEN_SIZE)
        self.block3 = ResidualBlock(HIDDEN_SIZE, HIDDEN_SIZE)
        self.block4 = ResidualBlock(HIDDEN_SIZE, HIDDEN_SIZE)
        self.block5 = ResidualBlock(HIDDEN_SIZE, HIDDEN_SIZE)
        self.block6 = LinearBlock(HIDDEN_SIZE, output_size, activation="tanh")
        if len(bound) == 1:
            self.bound = bound.tile((output_size))
        else:
            self.bound = bound.tile((int(output_size / 2),))

    def forward(self, x):
        if self.bound.device != x.device:
            self.bound = self.bound.to(x.device)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = x * self.bound
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, activation: str) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.batch_norm = nn.BatchNorm1d(output_size)
        if activation == "tanh":
            self.activation = nn.Tanh()
            self.linear.apply(self.init_weights_tanh)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
            self.linear.apply(self.init_weights_sigmoid)
        else:
            self.activation = nn.ReLU()
            self.linear.apply(self.init_weights_relu)

    def init_weights_relu(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            m.bias.data.fill_(0.01)

    def init_weights_tanh(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=5 / 3)
            m.bias.data.fill_(0.01)
    
    def init_weights_sigmoid(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1)
            m.bias.data.fill_(0.01)

    def forward(self, x, x0=None):
        x = self.linear(x)
        x = self.batch_norm(x)
        if x0 is None:
            x = self.activation(x)
        else:
            x = self.activation(x + x0)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, io_size, hidden_size):
        super().__init__()
        self.block1 = LinearBlock(io_size, hidden_size, activation="relu")
        self.block2 = LinearBlock(io_size, io_size, activation="relu")

    def forward(self, x0):
        x = self.block1(x0)
        x = self.block2(x, x0)
        return x

# different model
class MLPModelSmall(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.block1 = LinearBlock(input_size, 100, activation="relu")
        self.block2 = LinearBlock(100, output_size, activation="tanh")
        self.bound = torch.tensor([1, 0.8]).tile((int(output_size / 2),),)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x * self.bound
        return x
