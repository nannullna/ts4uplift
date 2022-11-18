import argparse

def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add training arguments to the parser"""
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default='saved/')
    parser.add_argument("--disable_wandb", action='store_true')
    parser.add_argument("--disable_tqdm", action='store_true')

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer_type", type=str, default="adamw")

    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=1)

    parser.add_argument("--alpha", type=float, default=0.2)

    return parser


def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add model arguments to the parser"""
    parser.add_argument("--model_type", type=str, default="siamese", choices=["siamese", "dragonnet"])
    parser.add_argument("--backbone_type", type=str, default="tcn", choices=["tcn", "lstm", "gru"])
    parser.add_argument("--pool_type", type=str, default="last")
    
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.2)

    return parser


def add_dataset_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add dataset arguments to the parser"""
    parser.add_argument("--dataset_path", type=str, nargs='+', required=True)
    parser.add_argument("--test_path", type=str, required=False)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    return parser