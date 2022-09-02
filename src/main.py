import torch
import numpy as np
import os

from config import config
from dataloader.loader import load_trace
from model.agent import central_agent, normal_agent

def main():
    random_seed = config["random_seed"]
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Create result dir
    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])

    # Create the queues for communication
    net_param_queue = []
    experience_queue = []
    for i in range(config["n_agent"]):
        net_param_queue.append(torch.multiprocessing.Queue(1))
        experience_queue.append(torch.multiprocessing.Queue(1))

    # Create agents
    coordinator = torch.multiprocessing.Process(
        target = central_agent,
        args = (net_param_queue, experience_queue),
    )
    coordinator.start()

    cooked_time_list, cooked_bandwidth_list, _, _ = load_trace()
    agent_list = []
    for i in range(config["n_agent"]):
        agent_list.append(torch.multiprocessing.Process(
            target = normal_agent,
            args = (
                net_param_queue[i], experience_queue[i],
                i, cooked_time_list, cooked_bandwidth_list,
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