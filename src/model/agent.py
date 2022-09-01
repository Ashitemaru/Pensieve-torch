import torch
import logging
import numpy as np

from config import config
from model.a3c import A3C

def central_agent(net_param_queue, experience_queue):
    """
    'net_param_queue' & 'experience_queue' is shared by central agent & other agents.
    """
    torch.set_num_threads(1) # TODO: Understand it!

    assert len(net_param_queue) == config["n_agent"]
    assert len(experience_queue) == config["n_agent"]

    # Set up logging
    logging.basicConfig(
        filename = config["log_dir"] + "/log_central",
        filemode = "w",
        level = logging.INFO,
    )

    # Construct net
    central_net = A3C(
        is_central = True,
        n_feature = [config["video_param_count"], config["past_video_chunk_num"]],
        n_action = len(config["video_bitrate"]),
    )

    for epoch in range(config["total_epoch"]):
        # Push the central agent params to all the normal agents
        central_actor_param = list(central_net.actor.parameters())
        for agent_id in range(config["n_agent"]):
            net_param_queue[agent_id].put(central_actor_param)

        total_reward = 0
        total_entropy = 0
        total_batch_len = 0

        # Pull training data from all the normal agents
        for agent_id in range(config["n_agent"]):
            state_batch, action_batch, reward_batch, _, info = experience_queue[agent_id].get()
            central_net.backward_gradients(state_batch, action_batch, reward_batch)

            total_reward += np.sum(reward_batch)
            total_entropy += np.sum(info["entropy"])
            total_batch_len += len(reward_batch)

        central_net.update_net()

        # Logging result
        avg_reward = total_reward / config["n_agent"]
        avg_entropy = total_entropy / total_batch_len
        logging.info(f"Epoch: {epoch}, AVG reward: {avg_reward}, AVG entropy: {avg_entropy}")

def normal_agent(
    net_param_queue,
    experience_queue,
    agent_id,
    cooked_time_list,
    cooked_bandwidth_list,
):
    """
    'net_param_queue' & 'experience_queue' is shared by central agent & other agents.
    """
    

if __name__ == "__main__":
    pass