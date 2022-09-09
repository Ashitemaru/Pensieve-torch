import numpy as np
import torch
import os
import argparse

from config import config
from dataloader.loader import load_trace
from model.agent import central_agent, normal_agent

def main():
    random_seed = config["random_seed"]
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    parser = argparse.ArgumentParser(description = "Parse entropy related args")
    parser.add_argument(
        "-i", "--init_entropy",
        required = False, type = float, dest = "init_entropy",
        default = config["default_entropy_weight"],
    )
    parser.add_argument(
        "-d", "--entropy_decay",
        required = False, type = float, dest = "entropy_decay",
        default = config["entropy_decay"],
    )
    parser.add_argument(
        "-e", "--epoch",
        required = False, type = int, dest = "total_epoch",
        default = config["total_epoch"],
    )
    parser.add_argument(
        "-c", "--checkpoint",
        required = False, type = int, dest = "checkpoint_epoch",
        default = config["checkpoint_epoch"],
    )
    args = parser.parse_args()

    # Create dir
    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])
    if not os.path.exists(config["model_dir"]):
        os.makedirs(config["model_dir"])
    if not os.path.exists(config["image_dir"]):
        os.makedirs(config["image_dir"])

    # Create the queues for communication
    net_param_queue = []
    experience_queue = []
    for i in range(config["n_agent"]):
        net_param_queue.append(torch.multiprocessing.Queue(1))
        experience_queue.append(torch.multiprocessing.Queue(1))

    # Create agents
    coordinator = torch.multiprocessing.Process(
        target = central_agent,
        args = (
            net_param_queue,
            experience_queue,
            args.init_entropy,
            args.entropy_decay,
            args.total_epoch,
            args.checkpoint_epoch,
        ),
    )
    coordinator.start()

    cooked_time_list, cooked_bandwidth_list, _ = load_trace(is_test = False)
    agent_list = []
    for i in range(config["n_agent"]):
        agent_list.append(torch.multiprocessing.Process(
            target = normal_agent,
            args = (
                net_param_queue[i], experience_queue[i],
                i, cooked_time_list, cooked_bandwidth_list,
                args.total_epoch,
            )
        ))
    for agent in agent_list:
        agent.start()

    # Wait for training
    coordinator.join()
    for agent in agent_list:
        agent.join()

if __name__ == "__main__":
    main()