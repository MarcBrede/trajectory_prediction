import torch
from torch import nn
import torch_geometric as tg
from itertools import permutations

HIDDEN_FEATURES = 100

class MLPBackbone(torch.nn.Module):
    def __init__(self, input_features) -> None:
        super(MLPBackbone, self).__init__()
        self.block0 = LinearBlock(input_features, HIDDEN_FEATURES)
        self.block1 = ResidualBlock(HIDDEN_FEATURES, HIDDEN_FEATURES)
        self.block2 = ResidualBlock(HIDDEN_FEATURES, HIDDEN_FEATURES)
        self.block3 = ResidualBlock(HIDDEN_FEATURES, HIDDEN_FEATURES)
        self.block4 = ResidualBlock(HIDDEN_FEATURES, HIDDEN_FEATURES)
    
    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)        
        return x

class GNNLayer(torch.nn.Module):
    def __init__(self) -> None:
        super(GNNLayer, self).__init__()
        self.block0 = ConvResidualBlock(HIDDEN_FEATURES, HIDDEN_FEATURES)
        self.block1 = ConvResidualBlock(HIDDEN_FEATURES, HIDDEN_FEATURES)
        self.block2 = ConvResidualBlock(HIDDEN_FEATURES, HIDDEN_FEATURES)
    
    def forward(self, x, edges):
        x, a_0 = self.block0(x, edges)
        x, a_1 = self.block1(x, edges)
        x, a_2 = self.block2(x, edges)
        return x, (a_0[0], (a_0[1] + a_1[1] + a_2[1]))

class Head(torch.nn.Module):
    def __init__(self, output_features, bound=None):
        super(Head, self).__init__()
        if bound is not None:
            self.block0 = LinearBlock(HIDDEN_FEATURES, output_features, activation="tanh")
            self.bound = torch.tensor(bound).tile((output_features))
        else:
            self.block0 = LinearBlock(HIDDEN_FEATURES, output_features, activation="softmax")
            self.bound = None

    def forward(self, x):
        x = self.block0(x)
        if self.bound is not None:
            x = x * self.bound
        return x

class GNNModel(torch.nn.Module):
    def __init__(self, input_features, output_features, max_num_vehicles, max_num_obstacles, bound=None):
        super(GNNModel, self).__init__()
        self.mlp = MLPBackbone(input_features)
        self.gnn = GNNLayer()
        self.head = Head(output_features, bound=bound)
        self.edge_template = generate_edge_template(5, 5)
    
    def forward(self, x, batches):
        marks = x[:, 0]
        vehicles = (marks == 0)
        obstacles = (marks == 1)
        edges_vehicles, edges_obstacles = get_edges(batches, self.edge_template)
        edges = torch.cat((edges_vehicles, edges_obstacles), dim=-1)
        x = self.mlp(x)
        x, a = self.gnn(x, edges)
        x = self.head(x)
        return x, x[vehicles], x[obstacles], edges_vehicles, edges_obstacles, a

class LinearBlock(torch.nn.Module):
    def __init__(self, in_node_num, out_node_num, activation="relu"):
        super().__init__()
        self.linear = tg.nn.Linear(in_node_num, out_node_num, weight_initializer='kaiming_uniform')
        self.bn = nn.BatchNorm1d(out_node_num)        
        if activation == "relu":
            self.a = nn.ReLU()
        elif activation == "tanh":
            self.a = nn.Tanh()
        elif activation == "softmax":
            self.a = nn.Softmax(dim=-1)
        else:
            raise NotImplementedError("Not implement this type of activation function!")
    
    def forward(self, x, x0=None):
        x = self.linear(x)
        x = self.bn(x)
        if x0 is not None:
            x=x+x0
        x = self.a(x)
        return x

class ConvResidualBlock(torch.nn.Module):
    def __init__(self, io_node_num, hidden_node_num):
        super().__init__()
        # self.conv1 = TransformerConv(io_node_num, hidden_node_num, heads=1, concat=False)
        self.conv1 = tg.nn.conv.TransformerConv(io_node_num, hidden_node_num, head=1, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden_node_num)
        self.a1 = nn.ReLU()
        # self.conv2 = TransformerConv(hidden_node_num, io_node_num, heads=1, concat=False)
        self.conv2 = tg.nn.conv.TransformerConv(io_node_num, hidden_node_num, head=1, concat=False)
        self.bn2 = nn.BatchNorm1d(io_node_num)
        self.a2 = nn.ReLU()
    
    def forward(self, x0, edges):
        x, a_1 = self.conv1(x0, edges, return_attention_weights=True)
        x = self.bn1(x)
        x = self.a1(x)
        x, a_2 = self.conv2(x, edges, return_attention_weights=True)
        x = self.bn2(x)
        x = x+x0
        x = self.a2(x)
        return x, (a_1[0], a_1[1] + a_2[1])

def generate_edge_template(max_num_vehicles, max_num_obstacles):
    
    assert max_num_vehicles >= 1, \
            'Must have at least one vehicle!'
    
    assert max_num_obstacles >= 0, \
            'Number of obstacle should be positive integer!'
            
    edge_template = {}
    
    for num_vehicles in range(1, max_num_vehicles + 1):
        for num_obstacles in range(max_num_obstacles + 1):
            
            edges_vehicles = torch.tensor([[],[]],dtype=torch.int)
            edges_obstacles = torch.tensor([[],[]],dtype=torch.int)
            
            if num_vehicles > 1:
                all_perm = list(permutations(range(num_vehicles), 2))
                vehicle_1, vehicle_2 = zip(*all_perm)
                vehicle_to_vehicle = torch.tensor([vehicle_1, vehicle_2])
                edges_vehicles = torch.cat((edges_vehicles, vehicle_to_vehicle),dim=-1)
            
            if num_obstacles > 0:
                obstacles = torch.arange(num_vehicles, num_vehicles+num_obstacles).tile(num_vehicles)
                vehicles = torch.arange(num_vehicles).repeat_interleave(num_obstacles)
                obstacle_to_vehicle = torch.cat((obstacles[None,:], vehicles[None,:]),dim=0)
                # vehicle_to_obstacle = torch.cat((vehicles[None,:], obstacles[None,:]),dim=0)
                edges_obstacles = torch.cat((edges_obstacles, 
                                                obstacle_to_vehicle, 
                                            #  vehicle_to_obstacle,
                                                ),dim=-1)
            
            edge_template[(num_vehicles+num_obstacles, num_vehicles)] = [edges_vehicles, edges_obstacles]
    
    return edge_template

class ResidualBlock(nn.Module):
    def __init__(self, io_size, hidden_size):
        super().__init__()
        self.block1 = LinearBlock(io_size, hidden_size, activation="relu")
        self.block2 = LinearBlock(io_size, io_size, activation="relu")

    def forward(self, x0):
        x = self.block1(x0)
        x = self.block2(x, x0)
        return x

def get_edges(batches, edge_template):            
    edges_vehicles = torch.tensor([[],[]],dtype=torch.int)
    edges_obstacles = torch.tensor([[],[]],dtype=torch.int)
    
    batches_offset = torch.cumsum(batches[:,0],dim=0)[:-1]
    batches_offset = torch.cat((torch.tensor([0]), batches_offset))
    
    for batch in torch.unique(batches, dim=0):
            
        index = torch.all(batches == batch, dim=-1)
        
        if torch.sum(index) == 0:
            continue
        
        offset = batches_offset[index]
        edges_batch_vehicles, edges_batch_obstacles = edge_template[tuple(batch.tolist())]
        
        edges_vehicles = torch.cat([edges_vehicles, (edges_batch_vehicles[:,None,:]+offset[None,:,None]).reshape(2,-1)], dim=-1)
        edges_obstacles = torch.cat([edges_obstacles, (edges_batch_obstacles[:,None,:]+offset[None,:,None]).reshape(2,-1)], dim=-1)
    
    return edges_vehicles, edges_obstacles


if __name__ == "__main__":
    print(tg.__version__)