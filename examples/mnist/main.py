import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import wandb


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train(epoch, run, model, dataloader, optimizer, loss_fn, log_interval=10):
    model.train()
    for i, (images, target) in enumerate(dataloader):
        images, target = images.to("mps"), target.to("mps")
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if i % log_interval:
            wandb.log({"train/loss": loss})


def main():
    num_epochs = 1
    batch_size = 64
    lr = 0.001

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )
    device = torch.device("mps")

    dataset = datasets.MNIST("/tmp", train=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    now = datetime.datetime.now()
    run_id = now.strftime("%Y-%m-%d-T%H%M%S")

    run = wandb.init(name=run_id, id=run_id, project="mnist", job_type="main")

    for epoch in range(1, num_epochs + 1):
        train(epoch, run, model, dataloader, optimizer, loss_fn)


if __name__ == "__main__":
    main()
