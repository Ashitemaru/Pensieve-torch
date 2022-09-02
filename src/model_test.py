import torch
import numpy as np
import os

from config import config
from dataloader.loader import load_trace
from env.environment import ABREnvironment
from model.network import ActorNet

# Helper variables
B_IN_KB = 1000
B_IN_MB = 1000000
KB_IN_MB = 1000
MILLISECOND_IN_SECOND = 1000

def model_test(actor_model_path, epoch):
    torch.set_num_threads(1)
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    cooked_time_list, \
    cooked_bandwidth_list, \
    cooked_file_name_list = load_trace(is_test = True)

    environment = ABREnvironment(cooked_time_list, cooked_bandwidth_list, is_fixed = True)

    # Set up logging
    if not os.path.exists(config["log_dir"] + f"/epoch_{epoch}_test"):
        os.makedirs(config["log_dir"] + f"/epoch_{epoch}_test")
    file_name = cooked_file_name_list[environment.random_trace_id]
    logging = open(
        config["log_dir"] + f"/epoch_{epoch}_test/{file_name}.log",
        "w"
    )
    logging.write(
        "Time Stamp\t" + "Bit Rate\t" +
        "Buffer Size\t" + "Rebuf Mark\t" +
        "Video Chunk Size\t" + "Delay\t" +
        "Reward\n"
    )

    actor_net = ActorNet(
        n_feature = [config["video_param_count"], config["past_video_chunk_num"]],
        n_action = len(config["video_bitrate"]),
    )
    actor_net.load_state_dict(torch.load(actor_model_path))

    # Start to interact with environment
    state = torch.zeros((config["video_param_count"], config["past_video_chunk_num"]))
    time_stamp = 0
    video_count = 0
    last_bitrate_level = config["default_bitrate_level"]
    bitrate_level = config["default_bitrate_level"]
    reward_map = {}
    while True:
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

        # Calc the reward
        # Based on the bitrate, then cut off all the penalty
        bitrate_list = config["video_bitrate"]
        reward = bitrate_list[bitrate_level] / KB_IN_MB \
            - config["rebuf_penalty"] * (1 if rebuf else 0) \
            - config["smooth_penalty"] * np.abs(
                bitrate_list[bitrate_level] - bitrate_list[last_bitrate_level]
            ) / KB_IN_MB
        
        # Log reward
        if file_name not in reward_map:
            reward_map[file_name] = [reward]
        else:
            reward_map[file_name].append(reward)

        last_bitrate_level = bitrate_level

        state = torch.roll(state, shifts = -1, dims = -1)
        buffer_norm = config["buffer_size_norm_factor"]
        total_chunk = config["total_video_chunk"]
        n_action = len(bitrate_list)
        state[0, -1] = bitrate_list[bitrate_level] / float(np.max(bitrate_list)) # Bitrate (normed)
        state[1, -1] = buffer_size / buffer_norm # Buffer size (normed, in 10 s)
        state[2, -1] = (float(video_chunk_size) / B_IN_KB) / float(delay) # Throughput (k-byte per ms)
        state[3, -1] = float(delay) / MILLISECOND_IN_SECOND / buffer_norm # Time (normed, in 10 s)
        state[4, : n_action] = torch.tensor(next_chunk_size_list) / B_IN_MB # Chunk size (in M-byte)
        state[5, -1] = np.minimum(video_chunk_left, total_chunk) / float(total_chunk)

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

        # Get next bitrate
        with torch.no_grad():
            action_prob = actor_net(state.unsqueeze(dim = 0))
            prob_generator = torch.distributions.Categorical(action_prob)
            bitrate_level = prob_generator.sample().item()

        if end_of_video:
            logging.write("========== END ==========\n")
            logging.close()

            # Reset all pointers
            last_bitrate_level = config["default_bitrate_level"]
            bitrate_level = config["default_bitrate_level"]
            state = torch.zeros((config["video_param_count"], config["past_video_chunk_num"]))
            video_count += 1

            if video_count >= len(cooked_file_name_list):
                break

            file_name = cooked_file_name_list[environment.random_trace_id]
            logging = open(
                config["log_dir"] + f"/epoch_{epoch}_test/{file_name}.log",
                "w"
            )
            logging.write(
                "Time Stamp\t" + "Bit Rate\t" +
                "Buffer Size\t" + "Rebuf Mark\t" +
                "Video Chunk Size\t" + "Delay\t" +
                "Reward\n"
            )

    # Reward logging
    reward_logging = open(config["log_dir"] + f"/epoch_{epoch}_reward.log", "w")
    total_reward = 0
    total_chunk = 0
    for file_name, reward_list in reward_map.items():
        reward_sum = sum(reward_list)
        chunk_count = len(reward_list)
        reward_logging.write(f"File: {file_name}, AVG Reward: {reward_sum / chunk_count}\n")
        total_reward += reward_sum
        total_chunk += chunk_count
    reward_logging.write(f"Total AVG Reward: {total_reward / total_chunk}\n")
    print(f"CHECKPOINT TEST OVER, Epoch: {epoch}, Total AVG Reward: {total_reward / total_chunk}\n")
    reward_logging.close()

    return total_reward / total_chunk

if __name__ == "__main__":
    pass