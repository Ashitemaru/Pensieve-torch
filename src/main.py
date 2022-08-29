import torch
import numpy as np
from config import config
import os
from model.a3c import A3C

def main():
    random_seed = config["random_seed"]
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    video_bitrate = config["video_bitrate"]
    n_action = len(video_bitrate)

    # Create result dir
    if not os.path.exists(config["summary_dir"]):
        os.makedirs(config["summary_dir"])

if __name__ == "__main__":
    main()