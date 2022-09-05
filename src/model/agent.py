import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import config
from env.environment import ABREnvironment
from model.a3c import A3C
from model_test import model_test

# Helper variables
B_IN_KB = 1000
B_IN_MB = 1000000
KB_IN_MB = 1000
MILLISECOND_IN_SECOND = 1000

# Global record
g_avg_reward_list = []

def central_agent(net_param_queue_list, experience_queue_list):
    """
    'net_param_queue_list' & 'experience_queue_list' is shared by central agent & other agents.
    """
    global g_avg_reward_list
    torch.set_num_threads(1) # TODO: Understand it!

    assert len(net_param_queue_list) == config["n_agent"]
    assert len(experience_queue_list) == config["n_agent"]

    # Set up logging
    logging.basicConfig(
        filename = config["log_dir"] + "/agent_central.log",
        filemode = "w",
        level = logging.INFO,
    )

    # Construct net
    central_net = A3C(
        is_central = True,
        n_feature = [config["video_param_count"], config["past_video_chunk_num"]],
        n_action = len(config["video_bitrate"]),
    )

    for epoch in tqdm(range(1, config["total_epoch"] + 1)):
        # Push the central agent params to all the normal agents
        central_actor_param = list(central_net.actor.parameters())
        for agent_id in range(config["n_agent"]):
            net_param_queue_list[agent_id].put(central_actor_param)

        total_reward = 0
        total_entropy = 0
        total_batch_len = 0

        # Pull training data from all the normal agents
        for agent_id in range(config["n_agent"]):
            state_batch, action_batch, reward_batch, _, info = experience_queue_list[agent_id].get()
            central_net.backward_gradients(state_batch, action_batch, reward_batch)

            total_reward += np.sum(reward_batch)
            total_entropy += np.sum(info["entropy"])
            total_batch_len += len(reward_batch)

        central_net.update_net()

        # Logging result
        avg_reward = total_reward / config["n_agent"]
        avg_entropy = total_entropy / total_batch_len
        logging.info(f"Epoch: {epoch}, AVG reward: {avg_reward}, AVG entropy: {avg_entropy}")

        # Checkpoint
        if epoch % config["checkpoint_epoch"] == 0:
            print(f"TRAIN DATASET CHECK, Epoch: {epoch}, AVG reward: {avg_reward}, AVG entropy: {avg_entropy}")
            torch.save(central_net.actor.state_dict(), config["model_dir"] + f"/actor_{epoch}.pt")
            # torch.save(central_net.critic.state_dict(), config["model_dir"] + f"/critic_{epoch}.pt")
            g_avg_reward_list.append(model_test(config["model_dir"] + f"/actor_{epoch}.pt", epoch))

        if epoch % config["img_checkpoint_epoch"] == 0:
            plt.figure()
            plt.plot(g_avg_reward_list)
            plt.xlabel(f"Epoch (divided by {config['checkpoint_epoch']})")
            plt.ylabel("Average Reward")
            plt.savefig(config["image_dir"] + f"/train_avg_reward_{epoch}.png")

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
    torch.set_num_threads(1)

    environment = ABREnvironment(
        cooked_time_list,
        cooked_bandwidth_list,
        is_fixed = False,
        random_seed = agent_id, # Use different seeds for every agent
    )

    with open(config["log_dir"] + f"/agent_{agent_id}.log", "w") as logging:
        logging.write(
            "Time Stamp\t" + "Bit Rate\t" +
            "Buffer Size\t" + "Rebuf Mark\t" +
            "Video Chunk Size\t" + "Delay\t" +
            "Reward\n"
        )

        agent_net = A3C(
            is_central = False,
            n_feature = [config["video_param_count"], config["past_video_chunk_num"]],
            n_action = len(config["video_bitrate"]),
        )

        time_stamp = 0
        for epoch in range(config["total_epoch"]):
            # Pull params from central agent
            actor_param = net_param_queue.get()
            agent_net.hard_update_actor(actor_param)

            # Register
            last_bitrate_level = config["default_bitrate_level"]
            bitrate_level = config["default_bitrate_level"]

            state_batch = []
            action_batch = []
            reward_batch = []
            entropy_record = []

            state = torch.zeros((1, config["video_param_count"], config["past_video_chunk_num"]))

            # Interact with the environment, get info
            delay, \
            sleep_time, \
            buffer_size, \
            rebuf, \
            video_chunk_size, \
            next_chunk_size_list, \
            end_of_video, \
            video_chunk_left = environment.get_video_chunk(bitrate_level)

            # Step the time stamp
            time_stamp += delay
            time_stamp += sleep_time

            while not end_of_video and len(state_batch) < config["train_batch_len"]:
                # Step in
                # 1. Bitrate reg step in
                last_bitrate_level = bitrate_level

                # 2. State roll up
                # Pop out the least recent video chunk
                state = state.clone().detach()
                state = torch.roll(state, shifts = -1, dims = -1)

                # Append a new video chunk
                bitrate_list = config["video_bitrate"]
                buffer_norm = config["buffer_size_norm_factor"]
                total_chunk = config["total_video_chunk"]
                n_action = len(bitrate_list)
                state[0, 0, -1] = bitrate_list[bitrate_level] / float(np.max(bitrate_list)) # Bitrate (normed)
                state[0, 1, -1] = buffer_size / buffer_norm # Buffer size (normed, in 10 s)
                state[0, 2, -1] = (float(video_chunk_size) / B_IN_KB) / float(delay) # Throughput (k-byte per ms)
                state[0, 3, -1] = float(delay) / MILLISECOND_IN_SECOND / buffer_norm # Time (normed, in 10 s)
                state[0, 4, : n_action] = torch.tensor(next_chunk_size_list) / B_IN_MB # Chunk size (in M-byte)
                # TODO: Why do this normalization?
                state[0, 5, -1] = np.minimum(video_chunk_left, total_chunk) / float(total_chunk)

                # Select a new bitrate
                bitrate_level = agent_net.select_action(state)

                # Interact with the environment, get info
                delay, \
                sleep_time, \
                buffer_size, \
                rebuf, \
                video_chunk_size, \
                next_chunk_size_list, \
                end_of_video, \
                video_chunk_left = environment.get_video_chunk(bitrate_level)

                # Calc the reward
                # Based on the bitrate, then cut off all the penalty
                reward = bitrate_list[bitrate_level] / KB_IN_MB \
                    - config["rebuf_penalty"] * (1 if rebuf else 0) \
                    - config["smooth_penalty"] * np.abs(
                        bitrate_list[bitrate_level] - bitrate_list[last_bitrate_level]
                    ) / KB_IN_MB

                # Add record
                state_batch.append(state)
                action_batch.append(bitrate_level)
                reward_batch.append(reward)
                entropy_record.append(0) # TODO: Add entropy record

                # Logging
                logging.write(
                    str(time_stamp) + "\t" +
                    str(bitrate_list[bitrate_level]) + "\t" +
                    str(buffer_size) + "\t" +
                    str(rebuf) + "\t" +
                    str(video_chunk_size) + "\t" +
                    str(delay) + "\t" +
                    str(reward) + "\n"
                )
                logging.flush()
            
            experience_queue.put([
                state_batch, action_batch, reward_batch,
                end_of_video,
                {
                    "entropy": entropy_record
                }
            ])
            logging.truncate(0)

if __name__ == "__main__":
    pass