import wandb
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from keras import models
from keras.optimizers import SGD, RMSprop, Adam, Adagrad
from engine import *


sweep_config = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters":
    {
        "num_layers": {"values": [1, 2, 3]},
        "num_neurons": {"values": [512, 1024, 2048, 4096]},
        "activation_func": {"values": ['relu', 'sigmoid', 'elu']},
        "learning_rate": {"values": [0.0001, 0.001, 0.01, 0.1]},
        "dropout": {"values": [0, 0.5]},
        "optimizer": {"values": ['sgd', 'adam']},
        "batch_size": {"value": 512},
        "epochs" : {"value" : 5},
    }
}


sweep_id = wandb.sweep(
    sweep=sweep_config,
    project="robotics-nn",
)


def main():
    run = wandb.init()

    num_layers = wandb.config.num_layers
    num_neurons = wandb.config.num_neurons
    activation_func = wandb.config.activation_func
    learning_rate = wandb.config.learning_rate
    dropout = wandb.config.dropout
    optimizer = wandb.config.optimizer
    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs

    set_seed(42)

    train_images, train_labels, _, _, val_images, val_labels = load_mnist_data()
    model = create_model(num_neurons, activation_func, num_layers, dropout, learning_rate, optimizer)
    train_losses, train_accs = train_model(model, train_images, train_labels, epochs, batch_size)
    val_loss, val_acc = evaluate_model(model, val_images, val_labels)

    for i, (train_loss, train_acc) in enumerate(zip(train_losses, train_accs)):
        wandb.log({
            "epoch": i,
            "train_acc": train_acc,
            "train_loss": train_loss,
        })

    wandb.log({
        "val_acc": val_acc,
        "val_loss": val_loss,
    })

wandb.agent(sweep_id, function=main)
