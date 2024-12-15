from math import ceil
import torch

# Token constants
BOS_TOKEN = -1  # Beginning of sequence token
EOS_TOKEN = -2  # End of sequence token

DIVISION_MODULO_OPERATIONS = {
    "x/y": lambda x, y, p: (x*y % p, y, x),
}

MULTIPLICATION_MODULO_OPERATIONS = {
    "x*y": lambda x, y, p: (x, y, (x * y) % p),
}

ALL_MODULO_OPERATIONS = {
    "x+y": lambda x, y, _: (x, y, x + y),
    "x-y": lambda x, y, _: (x, y, x - y),
    **DIVISION_MODULO_OPERATIONS,
    **MULTIPLICATION_MODULO_OPERATIONS,
}

ALL_OPERATIONS = {
    **ALL_MODULO_OPERATIONS,
}

def operation_mod_p_data(operation: str, p: int, eq_token: int, op_token: int):
    """
    Generate data for modular arithmetic operations with BOS and EOS tokens
    """
    x = torch.arange(0, p)
    y = torch.arange(0 if operation not in DIVISION_MODULO_OPERATIONS else 1, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    
    # Add BOS and EOS tokens
    bos = torch.ones_like(x) * (p + 2)  # BOS token
    eos = torch.ones_like(x) * (p + 3)  # EOS token

    x, y, labels = ALL_OPERATIONS[operation](x, y, p)

    # Stack with BOS at start and EOS at end
    inputs = torch.stack([bos, x, op, y, eq, eos], dim=1)

    return inputs, labels

def get_data(operation: str, prime: int, training_fraction: float, batch_size: int):
    inputs, labels = operation_mod_p_data(operation, prime, prime, prime+1)
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = min(batch_size, ceil(len(dataset) / 2))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader
