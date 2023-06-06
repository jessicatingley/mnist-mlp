import argparse
from engine import *


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('num_layers', type=int,)
    parser.add_argument('num_neurons', type=int,)
    parser.add_argument('activation_func', type=str,)
    parser.add_argument('learning_rate', type=float,)
    parser.add_argument('dropout', type=float,)
    parser.add_argument('batch_size', type=int,)
    parser.add_argument('optimizer', type=str,)
    parser.add_argument('epochs', type=int,)
    return parser


def main() -> None:
    set_seed(42)
    train_images, train_labels, test_images, test_labels, val_images, val_labels = load_mnist_data()
    parser = make_parser()
    args = parser.parse_args()
    model = create_model(args.num_neurons, args.activation_func, args.num_layers, args.dropout, args.learning_rate, args.optimizer)
    train_model(model, train_images, train_labels, args.epochs, args.batch_size,)
    val_loss, val_acc = evaluate_model(model, val_images, val_labels)
    test_loss, test_acc = evaluate_model(model, test_images, test_labels)
    print("val_loss:", val_loss, "\nval_acc:", val_acc)
    print("test_loss:", test_loss, "\ntest_acc:", test_acc)


if __name__ == "__main__":
    main()
