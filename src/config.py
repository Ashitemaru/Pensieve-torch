config = {
    "random_seed": 42,

    # Directory config
    "cooked_trace_dir": "../data/cooked_traces",
    "video_size_dir": "../data/video_size",
    "log_dir": "../log",

    # Video config
    "video_bitrate": [300, 750, 1200, 1850, 2850, 4300], # In Kbps
    "video_param_count": 6,
    "past_video_chunk_num": 8,
    "packet_payload_portion": 0.95,
    "link_round_trip_time": 80, # In ms
    "noise_lower_bound": 0.9,
    "noise_upper_bound": 1.1,
    "video_chunk_len": 4000, # In ms
    "buffer_threshold": 60000, # In ms
    "drain_buffer_sleep_time": 500, # In ms
    "total_video_chunk": 48,

    # RL config
    "actor_default_lr": 1e-4,
    "critic_default_lr": 1e-3,
    "default_reward_decay": 0.99,
    "default_entropy_weight": 0.5,
    "n_agent": 4,
    "total_epoch": 30000,

    # Network config
    "conv_kernel_size": 4,
    "n_conv_filter": 128,
    "first_fc_output_dim": 128,
    "hidden_fc_output_dim": 128,
}