config = {
    "random_seed": 42,

    # Directory config
    "summary_dir": "./summary",

    # Video config
    "video_bitrate": [300, 750, 1200, 1850, 2850, 4300], # In Kbps

    # RL config
    "actor_default_lr": 1e-4,
    "critic_default_lr": 1e-3,
    "default_reward_decay": 0.99,
    "default_entropy_weight": 0.5,
    "n_agent": 4,

    # Network config
    "conv_kernel_size": 4,
    "n_conv_filter": 128,
    "first_fc_output_dim": 128,
    "hidden_fc_output_dim": 128,
}