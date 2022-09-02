config = {
    "random_seed": 42,

    # Directory config
    "cooked_trace_dir": "../data/cooked_traces",
    "cooked_test_trace_dir": "../data/cooked_test_traces",
    "video_size_dir": "../data/video_size",
    "model_dir": "../model",
    "log_dir": "../log",
    "image_dir": "../image",

    # Video config
    "video_bitrate": [300, 750, 1200, 1850, 2850, 4300], # In k-byte per sec
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
    "default_bitrate_level": 1,
    "buffer_size_norm_factor": 10, # In sec

    # RL config
    "actor_default_lr": 1e-4,
    "critic_default_lr": 1e-3,
    "default_reward_decay": 0.99,
    "default_entropy_weight": 0.5,
    "n_agent": 4,
    "total_epoch": 30000,
    "checkpoint_epoch": 500,
    "train_batch_len": 100,

    # Reward config
    "rebuf_penalty": 4.3, # 1 sec rebuffer -> 3 M-byte per sec
    "smooth_penalty": 1,

    # Network config
    "conv_kernel_size": 4,
    "n_conv_filter": 128,
    "first_fc_output_dim": 128,
    "hidden_fc_output_dim": 128,
}