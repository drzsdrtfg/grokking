from math import ceil
import torch
import os
import json
import glob
import random

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
    # Apply modulo to ensure results are within [0, p-1] range
    "x+y": lambda x, y, p: (x, y, (x + y) % p),
    "x-y": lambda x, y, p: (x, y, (x - y) % p),
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

# TinyStories functions
def load_tinystories(data_path: str, min_length: int = 50):
    """
    Load TinyStories dataset with more robust handling
    
    Args:
        data_path: Path to the TinyStories data file or directory
        min_length: Minimum character length for a valid story
        
    Returns:
        List of stories
    """
    stories = []
    
    print(f"Loading TinyStories from {data_path}")
    
    # Case 1: Check if it's a directory
    if os.path.isdir(data_path):
        # Find all text files
        txt_files = glob.glob(os.path.join(data_path, "*.txt"))
        json_files = glob.glob(os.path.join(data_path, "*.json"))
        
        print(f"Found {len(txt_files)} txt files and {len(json_files)} json files")
        
        # Process text files
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if this is a single story or multiple stories separated by blank lines
                if "\n\n" in content and len(content) > 1000:
                    # This might be a collection file, split by double newlines
                    parts = [p.strip() for p in content.split("\n\n")]
                    for part in parts:
                        if len(part) >= min_length:
                            stories.append(part)
                else:
                    # Treat as a single story
                    if len(content) >= min_length:
                        stories.append(content)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        
        # Process JSON files
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            if 'text' in item and len(item['text']) >= min_length:
                                stories.append(item['text'])
                            elif 'story' in item and len(item['story']) >= min_length:
                                stories.append(item['story'])
                elif isinstance(data, dict):
                    if 'text' in data and len(data['text']) >= min_length:
                        stories.append(data['text'])
                    elif 'story' in data and len(data['story']) >= min_length:
                        stories.append(data['story'])
            except Exception as e:
                print(f"Error processing JSON file {file_path}: {e}")
    
    # Case 2: Check if it's a single file
    elif os.path.isfile(data_path):
        try:
            if data_path.endswith('.txt'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if this might be multiple stories
                if "\n\n" in content and len(content) > 1000:
                    parts = [p.strip() for p in content.split("\n\n")]
                    for part in parts:
                        if len(part) >= min_length:
                            stories.append(part)
                else:
                    if len(content) >= min_length:
                        stories.append(content)
                        
            elif data_path.endswith('.json'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            if 'text' in item and len(item['text']) >= min_length:
                                stories.append(item['text'])
                            elif 'story' in item and len(item['story']) >= min_length:
                                stories.append(item['story'])
                elif isinstance(data, dict):
                    if 'text' in data and len(data['text']) >= min_length:
                        stories.append(data['text'])
                    elif 'story' in data and len(data['story']) >= min_length:
                        stories.append(data['story'])
        except Exception as e:
            print(f"Error processing file {data_path}: {e}")
    
    # If we still don't have any stories, create some sample stories
    if len(stories) == 0:
        print("No stories found, creating sample stories...")
        stories = create_sample_stories(10)
    
    print(f"Loaded {len(stories)} stories")
    return stories

def create_sample_stories(num_stories=10):
    """Create simple sample stories if no real data is available"""
    sample_stories = [
        """Once upon a time, there was a little red bird. The bird loved to sing. Every morning, the bird sang a happy song. All the animals in the forest liked to hear the little bird sing. One day, the bird could not sing. It was very sad. A kind fox asked, "Why are you sad, little bird?" The bird said, "I lost my song!" The fox helped the bird look for its song. They looked high and low. Then, the bird drank some water. Suddenly, it could sing again! The little bird was very happy. It sang a special song for the fox. They became good friends.""",
        
        """Tim had a small toy car. The car was blue and fast. Tim played with his car every day. One day, Tim could not find his car. He looked under his bed. He looked in his toy box. He looked on his desk. The car was not there. Tim was sad. Then, his dog Max came into the room. Max had the blue car in his mouth! Tim laughed and said, "Max, that's my car!" Max gave the car back to Tim. Tim gave Max a big hug. Then they played together all day.""",
        
        """Lily wanted to plant a flower. She got a small pot and some soil. She put the soil in the pot. Then, she made a little hole in the soil. Lily put a seed in the hole and covered it with soil. Every day, Lily gave the pot some water. She put the pot near the window so it could get sun. Lily waited and waited. One week later, she saw a tiny green leaf! Lily was so happy. Her plant was growing! After many days, a beautiful purple flower bloomed. Lily showed everyone her pretty flower."""
    ]
    
    stories = []
    for _ in range(num_stories):
        # Pick a random story
        story = random.choice(sample_stories)
        
        # Simple modifications to create variations
        if random.random() < 0.5:
            # Change character names
            name_replacements = {
                "Tim": random.choice(["Tom", "Ben", "Max", "Leo"]),
                "Lily": random.choice(["Lucy", "Zoe", "Mia", "Anna"]),
                "Max": random.choice(["Rex", "Spot", "Buddy", "Ollie"])
            }
            
            for old_name, new_name in name_replacements.items():
                if old_name in story:
                    story = story.replace(old_name, new_name)
        
        if random.random() < 0.5:
            # Change colors
            color_replacements = {
                "red": random.choice(["blue", "green", "yellow", "purple"]),
                "blue": random.choice(["red", "green", "yellow", "purple"]),
                "purple": random.choice(["blue", "green", "red", "yellow"])
            }
            
            for old_color, new_color in color_replacements.items():
                if old_color in story:
                    story = story.replace(old_color, new_color)
        
        stories.append(story)
    
    return stories

def create_char_vocab(stories):
    """Create character-level vocabulary"""
    chars = set()
    for story in stories:
        chars.update(story)
    
    # Sort for deterministic ordering
    chars = sorted(list(chars))
    
    # Create vocabulary (char -> id)
    vocab = {char: i for i, char in enumerate(chars)}
    
    # Add special tokens
    vocab['<bos>'] = len(vocab)  # Beginning of sequence
    vocab['<eos>'] = len(vocab)  # End of sequence
    vocab['<pad>'] = len(vocab)  # Padding
    
    return vocab

def tokenize_story(story, vocab, seq_len):
    """Tokenize a story at character level"""
    # Convert to character tokens
    tokens = [vocab[c] for c in story if c in vocab]
    
    # Create sequences of length seq_len + 1 (input + target)
    sequences = []
    targets = []
    
    # Generate sequences with sliding window
    stride = max(1, seq_len // 4)  # Smaller stride = more training examples with overlap
    
    for i in range(0, len(tokens) - seq_len, stride):
        seq = tokens[i:i + seq_len]
        tgt = tokens[i + 1:i + seq_len + 1]
        
        # Pad if needed
        if len(seq) < seq_len:
            seq = seq + [vocab['<pad>']] * (seq_len - len(seq))
        if len(tgt) < seq_len:
            tgt = tgt + [vocab['<pad>']] * (seq_len - len(tgt))
        
        # Add BOS to beginning of sequence
        seq = [vocab['<bos>']] + seq[:-1]
        
        sequences.append(seq)
        targets.append(tgt)
    
    return sequences, targets

def get_tinystories_data(data_path, seq_len, batch_size, training_fraction=0.9):
    """Get data loaders for TinyStories dataset"""
    print(f"Loading TinyStories data from {data_path}...")
    stories = load_tinystories(data_path)
    
    if not stories:
        print("ERROR: No stories found or loaded")
        # Create synthetic stories as fallback
        stories = create_sample_stories(20)
    
    print(f"Creating vocabulary from {len(stories)} stories...")
    vocab = create_char_vocab(stories)
    print(f"Vocabulary size: {len(vocab)} characters")
    
    all_sequences = []
    all_targets = []
    
    print("Tokenizing stories...")
    for story in stories:
        sequences, targets = tokenize_story(story, vocab, seq_len)
        all_sequences.extend(sequences)
        all_targets.extend(targets)
    
    print(f"Created {len(all_sequences)} training sequences")
    
    # Ensure we have data
    if len(all_sequences) == 0:
        print("ERROR: No training sequences were generated")
        # Create a minimal dataset with random data so training can proceed
        print("Creating minimal synthetic dataset...")
        rand_seq = torch.randint(0, len(vocab), (100, seq_len))
        rand_tgt = torch.randint(0, len(vocab), (100, seq_len))
        
        inputs = rand_seq
        targets = rand_tgt
    else:
        # Convert to tensors
        inputs = torch.tensor(all_sequences)
        targets = torch.tensor(all_targets)
    
    print(f"Final dataset: {inputs.shape} inputs, {targets.shape} targets")
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    
    # Ensure we have enough data for both train and val
    min_val_size = 10  # Minimum number of validation samples
    if len(dataset) <= min_val_size:
        print(f"WARNING: Dataset too small ({len(dataset)} samples), creating minimal synthetic dataset")
        rand_seq = torch.randint(0, len(vocab), (100, seq_len))
        rand_tgt = torch.randint(0, len(vocab), (100, seq_len))
        dataset = torch.utils.data.TensorDataset(rand_seq, rand_tgt)
    
    # Split into train/val
    train_size = max(1, int(training_fraction * len(dataset)))
    val_size = max(1, len(dataset) - train_size)
    
    print(f"Splitting into {train_size} training and {val_size} validation samples")
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Adjust batch size if needed
    batch_size = min(batch_size, train_size // 2 or 1)
    print(f"Using batch size of {batch_size}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, vocab
