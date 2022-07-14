"""
A basic A + B linear model.
"""

import torch
from tqdm import tqdm

EPOCH = 10000
TEST_SIZE = 10000

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
    print("WARNING: Using CPU to train the model.")

class NaiveModel(torch.nn.Module):
    def __init__(self):
        super(NaiveModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

def train(model, optimizer):
    model.train()
    for _ in tqdm(range(EPOCH)):
        optimizer.zero_grad()

        x = torch.rand(2).to(device)
        loss = (model(x) - sum(x)) ** 2
        loss.backward()

        optimizer.step()

def evaluate(model):
    model.eval()
    total_loss = 0
    for _ in tqdm(range(TEST_SIZE)):
        x = torch.rand(2).to(device)
        loss = (model(x) - sum(x)) ** 2
        total_loss += loss.item()

    return total_loss

def main():
    # Init
    model = NaiveModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1)

    # Before training
    total_loss = evaluate(model)
    print(f"Before training, the average loss: {total_loss / TEST_SIZE}")
    for param in model.parameters():
        print(param)

    train(model, optimizer)

    # After training
    total_loss = evaluate(model)
    print(f"After training, the average loss: {total_loss / TEST_SIZE}")
    for param in model.parameters():
        print(param)

    print("PyTorch test passed!")

if __name__ == "__main__":
    main()