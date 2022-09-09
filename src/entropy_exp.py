import os
import matplotlib.pyplot as plt

from model_test import model_test
from config import config

def main():
    total_epoch = 10000
    
    print("Constant Entropy Experiment")
    avg_reward_list = []
    for i in range(21):
        entropy = i / 20
        decay = 1

        os.system(f"python main.py -i {entropy} -d {decay} -e {total_epoch + 10} -c {total_epoch}")

        avg_reward = model_test(f"../model/actor_{total_epoch}.pt", total_epoch)
        avg_reward_list.append(avg_reward)
        print(f"Init entropy: {entropy}, decay: {decay}, avg reward: {avg_reward}")

        os.system("rm -r ../log")
        os.system("rm -r ../model")

    plt.figure()
    plt.xlabel("Entropy")
    plt.ylabel("Average reward")
    plt.title("Avg reward in constant entropy weight")
    plt.plot([i / 20 for i in range(21)], avg_reward_list)
    plt.savefig(config["image_dir"] + "/constant_entropy.png")

    print("Decayed Entropy Experiment")
    avg_reward_list = []
    for i in range(10):
        entropy = 1
        decay_to = (i + 1) / 10
        decay = decay_to ** (1 / total_epoch)

        os.system(f"python main.py -i {entropy} -d {decay} -e {total_epoch + 10} -c {total_epoch}")

        avg_reward = model_test(f"../model/actor_{total_epoch}.pt", total_epoch)
        avg_reward_list.append(avg_reward)
        print(f"Init entropy: {entropy}, decay: {decay}, avg reward: {avg_reward}")

        os.system("rm -r ../log")
        os.system("rm -r ../model")

    plt.figure()
    plt.xlabel(f"Entropy decay to (after {total_epoch} epochs)")
    plt.ylabel("Average reward")
    plt.title("Avg reward in decayed entropy weight")
    plt.plot([(i + 1) / 10 for i in range(10)], avg_reward_list)
    plt.savefig(config["image_dir"] + "/decayed_entropy.png")

if __name__ == "__main__":
    main()