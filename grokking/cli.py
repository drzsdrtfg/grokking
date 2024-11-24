from argparse import ArgumentParser
from data import ALL_OPERATIONS
from model import Transformer  # Added missing import
from training import main

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--operation", type=str, choices=ALL_OPERATIONS.keys(), default="x*y")
    parser.add_argument("--training_fraction", type=float, default=0.2)
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--num_steps", type=int, default=1000000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(args)