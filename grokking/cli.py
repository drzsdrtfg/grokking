from argparse import ArgumentParser
from data import ALL_OPERATIONS
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
    parser.add_argument("--num_steps", type=int, default=17000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--inference_only", action="store_true", 
                    help="Run only inference using a saved model")
    parser.add_argument("--model_path", type=str, default=None,
                    help="Path to saved model for inference")
    args = parser.parse_args()

    main(args)
