import datetime

import torchvision
import wandb


now = datetime.datetime.now()
run_name = now.strftime("%Y-%m-%d-T%H%M%S")

run = wandb.init(
    name=run_name,
    id=run_name,
    project="mnist",
    job_type="dataset",
)

train_dataset = torchvision.datasets.MNIST(root="/tmp", train=True, download=True)
val_dataset = torchvision.datasets.MNIST(root="/tmp", train=False)

artifact = wandb.Artifact(name="mnist", type="dataset")
artifact.add_dir("/tmp/MNIST")

train_samples = wandb.Table(columns=["image", "label", "split"])
for image, label in train_dataset:
    train_samples.add_data(wandb.Image(image), label, "train")
artifact.add(train_samples, "train_samples")

val_samples = wandb.Table(columns=["image", "label", "split"])
for image, label in val_dataset:
    val_samples.add_data(wandb.Image(image), label, "val")
artifact.add(train_samples, "val_samples")

run.log_artifact(artifact)
