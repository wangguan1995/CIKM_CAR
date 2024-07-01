import sys
sys.path.append("./PaddleScience/")
sys.path.append('/home/aistudio/3rd_lib')
sys.path.append("./model")
import argparse
import os
import csv
from timeit import default_timer
from typing import List
import numpy as np
import paddle
import yaml
from paddle.optimizer.lr import LRScheduler
from src.data import instantiate_datamodule
from src.networks import instantiate_network
from src.utils.average_meter import AverageMeter
from src.utils.dot_dict import DotDict
from src.utils.dot_dict import flatten_dict


class StepDecay(LRScheduler):
    def __init__(
        self, learning_rate, step_size, gamma=0.1, last_epoch=-1, verbose=False
    ):
        if not isinstance(step_size, int):
            raise TypeError(
                "The type of 'step_size' must be 'int', but received %s."
                % type(step_size)
            )
        if gamma >= 1.0:
            raise ValueError("gamma should be < 1.0.")

        self.step_size = step_size
        self.gamma = gamma
        super().__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        i = self.last_epoch // self.step_size
        return self.base_lr * (self.gamma**i)


def instantiate_scheduler(config):
    if config.opt_scheduler == "CosineAnnealingLR":
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            config.lr, T_max=config.opt_scheduler_T_max
        )
    elif config.opt_scheduler == "StepLR":
        scheduler = StepDecay(
            config.lr, step_size=config.opt_step_size, gamma=config.opt_gamma
        )
    else:
        raise ValueError(f"Got {config.opt_scheduler=}")
    return scheduler


# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * paddle.norm(
            x.reshape((num_examples, -1)) - y.reshape((num_examples, -1)), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return paddle.mean(all_norms)
            else:
                return paddle.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        diff_norms = paddle.norm(x-y, 2)
        y_norms = paddle.norm(y, self.p)

        if self.reduction:
            if self.size_average:
                return paddle.mean(diff_norms / y_norms)
            else:
                return paddle.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def set_seed(seed: int = 0):
    paddle.seed(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)


def str2intlist(s: str) -> List[int]:
    return [int(item.strip()) for item in s.split(",")]


def parse_args(yaml="UnetShapeNetCar.yaml"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/"+ yaml,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file to resume training",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="log",
        help="Path to the log directory",
    )
    parser.add_argument("--logger_types", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for training")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument(
        "--sdf_spatial_resolution",
        type=str2intlist,
        default=None,
        help="SDF spatial resolution. Use comma to separate the values e.g. 32,32,32.",
    )

    args, _ = parser.parse_known_args()
    return args


def load_config(config_path):
    def include_constructor(loader, node):
        # Get the path of the current YAML file
        current_file_path = loader.name

        # Get the folder containing the current YAML file
        base_folder = os.path.dirname(current_file_path)

        # Get the included file path, relative to the current file
        included_file = os.path.join(base_folder, loader.construct_scalar(node))

        # Read and parse the included file
        with open(included_file, "r") as file:
            return yaml.load(file, Loader=yaml.Loader)

    # Register the custom constructor for !include
    yaml.Loader.add_constructor("!include", include_constructor)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Convert to dot dict
    config_flat = flatten_dict(config)
    config_flat = DotDict(config_flat)
    return config_flat

import re
 
def extract_numbers(s):
    return [int(digit) for digit in re.findall(r'\d+', s)]


def write_to_vtk(out_dict, point_data_pos="press on mesh points", mesh_path=None, track=None):
    import meshio
    p = out_dict["pressure"]
    index = extract_numbers(mesh_path.name)[0]

    if track == "Dataset_1":
        index = str(index).zfill(3)   
    elif track == "Track_B":
        index = str(index).zfill(4)

    print(f"Pressure shape for mesh {index} = {p.shape}")

        
    if point_data_pos == "press on mesh points":
        mesh = meshio.read(mesh_path)
        mesh.point_data["p"] = p.numpy()
        if "pred wss_x" in out_dict:
            wss_x = out_dict["pred wss_x"]
            mesh.point_data["wss_x"] = wss_x.numpy()
    elif point_data_pos == "press on mesh cells":
        points = np.load(mesh_path.parent / f"centroid_{index}.npy")
        npoint = points.shape[0]
        mesh = meshio.Mesh(
            points=points, cells=[("vertex", np.arange(npoint).reshape(npoint, 1))]
        )
        mesh.point_data = {"p":p.numpy()}

    print(f"write : ./output/{mesh_path.parent.name}_{index}.vtk")
    mesh.write(f"./output/{mesh_path.parent.name}_{index}.vtk") 
    

@paddle.no_grad()
def eval(model, datamodule, config, loss_fn=None, track="Dataset_1"):
    os.makedirs(f"./output/{track}", exist_ok=True)
    test_loader = datamodule.test_dataloader(batch_size=config.eval_batch_size, shuffle=False, num_workers=0)
    data_list = []

    for i, data_dict in enumerate(test_loader):
        print(data_dict.keys())
        out_dict = model.eval_dict(data_dict, loss_fn=loss_fn, decode_fn=datamodule.decode)
        if'l2 eval loss' in out_dict: 
            if i == 0:
                data_list.append(['id', 'l2 p'])
            else:
                data_list.append([i, float(out_dict['l2 eval loss'])])
        
        if config.write_to_vtk is True:
            print("datamodule.test_mesh_paths = ", datamodule.test_mesh_paths[i])
            write_to_vtk(out_dict, config.point_data_pos, datamodule.test_mesh_paths[i], track)
        
        # Your submit your npy to leaderboard here
        if "pressure" in out_dict:
            p = out_dict["pressure"].reshape((-1,)).astype(np.float32)
            test_indice = datamodule.test_indices[i]
            npy_leaderboard = f"./output/{track}/press_{str(test_indice).zfill(3)}.npy"
            print(f"saving *.npy file for [{track}] leaderboard : ", npy_leaderboard)
            np.save(npy_leaderboard, p)
        
        if "velocity" in out_dict:
            v = out_dict["velocity"].reshape((-1,3)).astype(np.float32)
            test_indice = datamodule.test_indices[i]
            npy_leaderboard = f"./output/{track}/vel_{str(test_indice).zfill(3)}.npy"
            print(f"saving *.npy file for [{track}] leaderboard : ", npy_leaderboard)
            np.save(npy_leaderboard, v)

        # check csv in ./output
        with open(f"./output/{config.project_name}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data_list)
            
    return

def train(config):
    model = instantiate_network(config)
    datamodule = instantiate_datamodule(config)
    train_loader = datamodule.train_dataloader(batch_size=config.batch_size, shuffle=False)

    # Initialize the optimizer
    scheduler = instantiate_scheduler(config)
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=scheduler, weight_decay=1e-4
    )

    # Initialize the loss function
    loss_fn = LpLoss(size_average=True)
    L2 = []
    for ep in range(config.num_epochs):
        model.train()
        t1 = default_timer()
        train_l2_meter = AverageMeter()
        # train_reg = 0
        for i, data_dict in enumerate(train_loader):
            optimizer.clear_grad()
            loss_dict = model.loss_dict(data_dict, loss_fn=loss_fn)
            loss = 0
            for k, v in loss_dict.items():
                loss = loss + v.mean()
            loss.backward()
            optimizer.step()
            train_l2_meter.update(loss.item())
        scheduler.step()
        t2 = default_timer()
        print(
            f"Training epoch {ep} took {t2 - t1:.2f} seconds. L2 loss: {train_l2_meter.avg:.4f}"
        )

        L2.append(train_l2_meter.avg)
        if ep % config.eval_interval == 0 or ep == config.num_epochs - 1:
            eval_dict = eval(model, datamodule, config, loss_fn)
            if eval_dict is not None:
                for k, v in eval_dict.items():
                    print(f"Epoch: {ep} {k}: {v.item():.4f}")
        # Save the weights
        if ep % config.save_interval == 0 or ep == config.num_epochs - 1 and ep > 1:
            paddle.save(
                model.state_dict(),
                os.path.join("./output/", f"model-{config.model}-{config.track}-{config.model}-{ep}.pdparams"),
            )

def load_yaml(file_name):
    args = parse_args(file_name)
    # args = parse_args("Unet_Velocity.yaml")
    config = load_config(args.config)

    # Update config with command line arguments
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value

    # pretty print the config
    if paddle.distributed.get_rank() == 0:
        print(f"\n--------------- Config [{file_name}] Table----------------")
        for key, value in config.items():
            print("Key: {:<30} Val: {}".format(key, value))
        print("--------------- Config yaml Table----------------\n")
    return config

def leader_board(track):
    track = config.track
    model = instantiate_network(config)
    checkpoint = paddle.load(f"./output/model-{config.model}-{track}-{config.num_epochs - 1}.pdparams")
    model.load_dict(checkpoint)
    print(f"\n-------Starting Evaluation over [{track}] --------")
    config.n_train = 1
    t1 = default_timer()
    os.makedirs(f"./output/{track}", exist_ok=True)
    config.mode="test"
    config.n_test=10
    eval_dict = eval(
        model, instantiate_datamodule(config), config, loss_fn=lambda x,y:0, track=track
    )
    t2 = default_timer()
    print(f"Inference over [Dataset_1 pressure] took {t2 - t1:.2f} seconds.")
    os.makedirs(f"./output/{track}", exist_ok=True)
    os.system(f"zip -r ./output/{track}.zip ./output/{track}")

if __name__ == "__main__":
    os.makedirs("./output/", exist_ok=True)
    
    config = load_yaml("UnetShapeNetCar.yaml")
    train(config)
    
    index_list = np.loadtxt("/home/aistudio/data/train_data_1_velocity/watertight_meshes.txt", dtype=int)
    config = load_yaml("Unet_Velocity.yaml")
    config.train_index_list = index_list[:10].tolist()  
    config.test_index_list = index_list[10:11].tolist()  
    train(config)

    config = load_yaml("Unet_Cd.yaml")
    index_list = np.loadtxt("/home/aistudio/data/train_data_2_cd/DataSet2.csv", delimiter=",", dtype=str)[:,1][1:10]
    config.train_index_list = index_list[:5].tolist()  
    config.test_index_list = index_list[5:6].tolist()  
    train(config)
    
    leader_board("Dataset_1_pressure")
    leader_board("Dataset_1_velocity")
    