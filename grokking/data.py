import os
import json
import torch
from math import ceil
import glob
import random

# Token constants
BOS_TOKEN = 133  # Beginning of sequence token
EOS_TOKEN = 134  # End of sequence token
PAD_TOKEN = 135  # Padding token

# Define the shared vocabulary structure
SHARED_VOCAB = {
    # Tab, Newline, Space
    '\t': 0, '\n': 1, ' ': 2,
    
    # Punctuation
    '!': 3, '"': 4, '#': 5, '$': 6, '&': 7, "'": 8, '(': 9, ')': 10,
    '*': 11, '+': 12, ',': 13, '-': 14, '.': 15, '/': 16,
    
    # Digits
    '0': 17, '1': 18, '2': 19, '3': 20, '4': 21, 
    '5': 22, '6': 23, '7': 24, '8': 25, '9': 26,
    
    # More punctuation
    ':': 27, ';': 28, '<': 29, '=': 30, '>': 31, '?': 32,
    
    # Uppercase letters
    'A': 33, 'B': 34, 'C': 35, 'D': 36, 'E': 37, 'F': 38, 'G': 39, 'H': 40,
    'I': 41, 'J': 42, 'K': 43, 'L': 44, 'M': 45, 'N': 46, 'O': 47, 'P': 48,
    'Q': 49, 'R': 50, 'S': 51, 'T': 52, 'U': 53, 'V': 54, 'W': 55, 'X': 56, 
    'Y': 57, 'Z': 58,
    
    # More punctuation
    '[': 59, '\\': 60, ']': 61, '_': 62, '`': 63,
    
    # Lowercase letters
    'a': 64, 'b': 65, 'c': 66, 'd': 67, 'e': 68, 'f': 69, 'g': 70, 'h': 71,
    'i': 72, 'j': 73, 'k': 74, 'l': 75, 'm': 76, 'n': 77, 'o': 78, 'p': 79,
    'q': 80, 'r': 81, 's': 82, 't': 83, 'u': 84, 'v': 85, 'w': 86, 'x': 87,
    'y': 88, 'z': 89,
    
    # Other characters and special tokens
    '~': 90, '\x92': 91, '\xa0': 92, '§': 93, '\xad': 94, '´': 95, '·': 96,
    'é': 97, 'í': 98, 'ï': 99, 'ñ': 100, 'ö': 101, 'İ': 102, 'ɪ': 103, 'ʏ': 104,
    'ʙ': 105, 'ʜ': 106, 'ғ': 107, 'ᴀ': 108, 'ᴄ': 109, 'ᴅ': 110, 'ᴇ': 111, 'ᴏ': 112,
    'ᴛ': 113, 'ᴜ': 114, 'ᴡ': 115, 'ᴢ': 116, '\u2005': 117, '\u2009': 118, '\u200a': 119,
    '\u200b': 120, '‑': 121, '–': 122, '—': 123, '―': 124, ''': 125, ''': 126, '"': 127,
    '"': 128, '„': 129, '…': 130, '\u2028': 131, '\u3000': 132,
    
    # Special tokens
    '<bos>': BOS_TOKEN,
    '<eos>': EOS_TOKEN,
    '<pad>': PAD_TOKEN
}

# Create reverse mapping
SHARED_VOCAB_REVERSE = {v: k for k, v in SHARED_VOCAB.items()}

# Total vocabulary size including special tokens
VOCAB_SIZE = len(SHARED_VOCAB)

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

def operation_mod_p_data(operation: str, p: int):
    """
    Generate data for modular arithmetic operations with shared vocabulary tokens
    """
    # Ensure p is not greater than our vocabulary size
    assert p <= 128, f"Prime modulus must be <= 128, got {p}"
    
    # Generate all possible x, y pairs
    x = torch.arange(0, p)
    y = torch.arange(0 if operation not in DIVISION_MODULO_OPERATIONS else 1, p)
    x, y = torch.cartesian_prod(x, y).T
    
    # Map operators and symbols to vocab tokens
    op_str = operation[1]  # Extract operation symbol (*, +, -, /)
    op_token = SHARED_VOCAB.get(op_str, SHARED_VOCAB['*'])  # Default to * if not found
    eq_token = SHARED_VOCAB['=']
    
    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    
    # Add BOS and EOS tokens
    bos = torch.ones_like(x) * BOS_TOKEN
    eos = torch.ones_like(x) * EOS_TOKEN
    
    # Generate labels
    x_vals, y_vals, labels = ALL_OPERATIONS[operation](x, y, p)
    
    # Stack with BOS at start and EOS at end
    inputs = torch.stack([bos, x_vals, op, y_vals, eq, eos], dim=1)
    
    return inputs, labels

def get_data(operation: str, prime: int, training_fraction: float, batch_size: int):
    """Get data for arithmetic operations using shared vocabulary"""
    inputs, labels = operation_mod_p_data(operation, prime)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    
    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    batch_size = min(batch_size, ceil(len(dataset) / 2))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader

def load_shared_vocab(vocab_path=None):
    """
    Load the shared vocabulary or create it if it doesn't exist
    """
    if vocab_path and os.path.exists(vocab_path):
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            print(f"Error loading vocabulary from {vocab_path}, using default")
    
    return SHARED_VOCAB

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

def tokenize_story(story, vocab, seq_len):
    """Tokenize a story using the shared vocabulary"""
    # Convert to character tokens, replacing unknown chars with space
    tokens = [vocab.get(c, vocab[' ']) for c in story]
    
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
    """Get data loaders for TinyStories dataset with shared vocabulary using streaming approach"""
    print(f"Loading TinyStories data from {data_path}...")
    
    # Load or use default shared vocabulary
    vocab_path = os.path.join(os.path.dirname(data_path) if os.path.isfile(data_path) else data_path, "vocabulary.json")
    vocab = load_shared_vocab(vocab_path)
    
    print(f"Using shared vocabulary with {len(vocab)} tokens")
    
    # Create a custom dataset that loads and tokenizes stories on-demand
    class StreamingTinyStoriesDataset(torch.utils.data.Dataset):
        def __init__(self, data_path, vocab, seq_len):
            self.data_path = data_path
            self.vocab = vocab
            self.seq_len = seq_len
            self.story_files = []
            self.story_sizes = {}  # Store file sizes for better estimation
            self.avg_chars_per_seq = seq_len * 4  # Initial estimate
            
            # Index available files instead of loading all content
            if os.path.isdir(data_path):
                self.story_files = (
                    glob.glob(os.path.join(data_path, "*.txt")) + 
                    glob.glob(os.path.join(data_path, "*.json"))
                )
                
                # Get file sizes for all files (more accurate estimation)
                for file_path in self.story_files:
                    try:
                        self.story_sizes[file_path] = os.path.getsize(file_path)
                    except:
                        self.story_sizes[file_path] = 1000  # Default size if can't determine
            
            elif os.path.isfile(data_path):
                self.story_files = [data_path]
                try:
                    self.story_sizes[data_path] = os.path.getsize(data_path)
                except:
                    self.story_sizes[data_path] = 1000  # Default size
            
            # If no files found, create a small set of samples
            if not self.story_files:
                print("No files found, creating sample stories...")
                self.sample_stories = create_sample_stories(10)
                self.sequences = []
                self.targets = []
                
                # Pre-tokenize the small sample set
                for story in self.sample_stories:
                    seq, tgt = self._tokenize_story(story)
                    self.sequences.extend(seq)
                    self.targets.extend(tgt)
                
                self.length = len(self.sequences)
            else:
                # Check if the dataset provides statistics
                stats_file = os.path.join(os.path.dirname(data_path) if os.path.isfile(data_path) else data_path, "stats.json")
                if os.path.exists(stats_file):
                    try:
                        with open(stats_file, 'r') as f:
                            stats = json.load(f)
                            # Use statistics to estimate dataset size more accurately
                            if 'total_characters' in stats and 'total_stories' in stats:
                                total_chars = stats['total_characters']
                                self.avg_chars_per_seq = stats.get('average_story_length', seq_len * 4)
                                
                                # Calculate based on actual story statistics if available
                                stride = max(1, self.seq_len // 4)
                                sequences_per_story = max(1, (self.avg_chars_per_seq - self.seq_len) / stride + 1)
                                total_seqs = int(stats['total_stories'] * sequences_per_story)
                                print(f"Dataset statistics from stats.json: {stats['total_stories']} stories, {total_chars} characters")
                                print(f"Estimated sequences per story: {sequences_per_story:.1f}")
                                print(f"Estimated total sequences: {total_seqs}")
                                self.length = total_seqs
                                return
                    except Exception as e:
                        print(f"Could not use stats file: {e}")
                
                # Estimate based on file sizes
                self.length = self._estimate_dataset_size()
        
        def _estimate_dataset_size(self):
            """Estimate dataset size more accurately based on file sizes and sampling"""
            if not self.story_files:
                return 100  # Default minimum size
                
            # Calculate total bytes in dataset
            total_bytes = sum(self.story_sizes.values())
            
            # Sample a few files to estimate sequences per byte
            sample_size = min(10, len(self.story_files))
            sampled_files = random.sample(self.story_files, sample_size) if sample_size > 0 else []
            
            total_chars = 0
            total_seqs = 0
            
            for file_path in sampled_files:
                try:
                    # Read file content (up to 50KB to avoid memory issues)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        sample = f.read(50000)
                    
                    if sample:
                        # Count characters
                        chars_in_sample = len(sample)
                        total_chars += chars_in_sample
                        
                        # Estimate sequences
                        sequences, _ = self._tokenize_story(sample)
                        seqs_in_sample = len(sequences)
                        total_seqs += seqs_in_sample
                        
                        # Update average chars per sequence estimate
                        if seqs_in_sample > 0:
                            chars_per_seq = chars_in_sample / seqs_in_sample
                            self.avg_chars_per_seq = (self.avg_chars_per_seq + chars_per_seq) / 2
                except Exception as e:
                    print(f"Error sampling file {file_path}: {e}")
            
            # Calculate sequences per byte
            if total_chars > 0:
                seqs_per_byte = total_seqs / total_chars
                
                # Extrapolate to entire dataset
                estimated_total = int(total_bytes * seqs_per_byte)
                
                # Apply scaling factor for safety
                safe_estimate = max(10000, estimated_total)
                
                print(f"Dataset estimation: {len(self.story_files)} files, ~{total_bytes:,} bytes")
                print(f"Sampling found {total_seqs} sequences in {total_chars:,} characters")
                print(f"Estimated total sequences: {safe_estimate:,}")
                
                return safe_estimate
            
            # Fallback: use simple heuristic based on file count and average size
            file_count = len(self.story_files)
            estimated_stories = file_count
            stride = max(1, self.seq_len // 4)
            
            # Estimate based on average story length and stride
            sequences_per_story = max(1, (self.avg_chars_per_seq - self.seq_len) / stride + 1)
            estimated_total = int(estimated_stories * sequences_per_story)
            
            print(f"Fallback estimation: {file_count} files")
            print(f"Estimated sequences per story: {sequences_per_story:.1f}")
            print(f"Estimated total sequences: {estimated_total:,}")
            
            return max(10000, estimated_total)  # Ensure a decent minimum
        
        def _get_file_for_index(self, idx):
            """Deterministically map an index to a file and position within the file"""
            if not self.story_files:
                return None, 0
                
            # Map idx to file using a deterministic pattern that spreads across files
            # This helps ensure we get good coverage of the dataset even with small batches
            file_idx = (idx * 104729) % len(self.story_files)  # Use prime number for better distribution
            file_path = self.story_files[file_idx]
            
            # Map to a position within the file (for files with multiple stories)
            # Different indices that map to the same file should access different stories
            position = (idx // len(self.story_files)) % 10  # 10 different positions per file
            
            return file_path, position
        
        def _load_story_batch(self, idx):
            """Load a batch of stories based on index"""
            # If we're using sample stories, return from pre-tokenized data
            if not self.story_files:
                if hasattr(self, 'sequences') and self.sequences:
                    return self.sequences[idx % len(self.sequences)], self.targets[idx % len(self.targets)]
                else:
                    # Create a minimal valid sequence as fallback
                    seq = [self.vocab['<bos>']] + [self.vocab[' ']] * (self.seq_len - 1)
                    tgt = [self.vocab[' ']] * self.seq_len
                    return seq, tgt
            
            # Determine which file to load based on index
            file_path, position = self._get_file_for_index(idx)
            
            try:
                stories = []
                if file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check if this might be multiple stories
                    parts = []
                    if "\n\n" in content and len(content) > 1000:
                        # Split by double newlines
                        parts = [p.strip() for p in content.split("\n\n")]
                    else:
                        # Treat as a single story
                        parts = [content]
                    
                    # Filter valid stories
                    for part in parts:
                        if len(part) >= 50:  # min_length
                            stories.append(part)
                            
                elif file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                if 'text' in item and len(item['text']) >= 50:
                                    stories.append(item['text'])
                                elif 'story' in item and len(item['story']) >= 50:
                                    stories.append(item['story'])
                    elif isinstance(data, dict):
                        if 'text' in data and len(data['text']) >= 50:
                            stories.append(data['text'])
                        elif 'story' in data and len(data['story']) >= 50:
                            stories.append(data['story'])
                
                # If we got stories, tokenize the appropriate one based on position
                if stories:
                    # Select a story based on position within the file
                    story_idx = position % len(stories)
                    story = stories[story_idx]
                    
                    # Tokenize this specific story
                    sequences, targets = self._tokenize_story(story)
                    
                    # Select a specific sequence from this story based on another transform of idx
                    # This helps ensure we get different sequences from the same story
                    if sequences:
                        seq_idx = (idx * 37) % max(1, len(sequences))  # Use another prime for distribution
                        return sequences[seq_idx % len(sequences)], targets[seq_idx % len(sequences)]
            
            except Exception as e:
                print(f"Error processing file {file_path} at idx {idx}: {e}")
            
            # Fallback: return a sequence from sample stories
            if not hasattr(self, 'sample_stories'):
                self.sample_stories = create_sample_stories(3)
            
            story = random.choice(self.sample_stories)
            sequences, targets = self._tokenize_story(story)
            if sequences:
                return sequences[0], targets[0]
            else:
                # Ultimate fallback - create a minimal valid sequence
                seq = [self.vocab['<bos>']] + [self.vocab[' ']] * (self.seq_len - 1)
                tgt = [self.vocab[' ']] * self.seq_len
                return seq, tgt
        
        def _tokenize_story(self, story):
            """Tokenize a single story"""
            # Convert to character tokens, replacing unknown chars with space
            tokens = [self.vocab.get(c, self.vocab[' ']) for c in story]
            
            # Create sequences of length seq_len + 1 (input + target)
            sequences = []
            targets = []
            
            # Generate sequences with sliding window
            stride = max(1, self.seq_len // 4)
            
            for i in range(0, len(tokens) - self.seq_len, stride):
                seq = tokens[i:i + self.seq_len]
                tgt = tokens[i + 1:i + self.seq_len + 1]
                
                # Pad if needed
                if len(seq) < self.seq_len:
                    seq = seq + [self.vocab['<pad>']] * (self.seq_len - len(seq))
                if len(tgt) < self.seq_len:
                    tgt = tgt + [self.vocab['<pad>']] * (self.seq_len - len(tgt))
                
                # Add BOS to beginning of sequence
                seq = [self.vocab['<bos>']] + seq[:-1]
                
                sequences.append(seq)
                targets.append(tgt)
            
            # Ensure we always return at least one sequence
            if not sequences:
                # Create a simple padded sequence
                seq = [self.vocab['<bos>']] + [self.vocab[' ']] * (self.seq_len - 1)
                tgt = [self.vocab[' ']] * self.seq_len
                sequences.append(seq)
                targets.append(tgt)
                
            return sequences, targets
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            """Get a single tokenized sequence by index"""
            seq, tgt = self._load_story_batch(idx % self.length)  # Use modulo to handle any index
            return torch.tensor(seq), torch.tensor(tgt)
    
    # Create streaming dataset
    dataset = StreamingTinyStoriesDataset(data_path, vocab, seq_len)
    
    # Define minimum sizes for train/val
    min_train_size = 30
    min_val_size = 10
    total_size = len(dataset)
    
    # Calculate train/val split sizes
    train_size = max(min_train_size, int(training_fraction * total_size))
    val_size = max(min_val_size, total_size - train_size)
    
    print(f"Dataset split: {train_size:,} training and {val_size:,} validation samples")
    
    # Create indices for train/val split
    # Instead of shuffling all indices (which could be billions), 
    # use a deterministic but seemingly random mapping
    class DeterministicRandomSampler(torch.utils.data.Sampler):
        def __init__(self, start_idx, end_idx, seed=42):
            self.start = start_idx
            self.end = end_idx
            self.size = end_idx - start_idx
            self.seed = seed
            
        def __iter__(self):
            # Use a prime-based mapping for pseudo-randomness
            for i in range(self.size):
                # Map i to a seemingly random but deterministic index in our range
                # Using modular arithmetic with large primes
                idx = (self.start + ((i * 104729 + 7919) % self.size)) % self.end
                yield idx
                
        def __len__(self):
            return self.size
    
    # Create samplers using deterministic random mapping
    train_sampler = DeterministicRandomSampler(0, train_size)
    val_sampler = DeterministicRandomSampler(train_size, train_size + val_size, seed=99)
    
    # Adjust batch size if needed (but allow larger batches for efficiency)
    batch_size = min(batch_size, train_size // 2)
    print(f"Using batch size of {batch_size}")
    
    # Create data loaders with samplers
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=0, pin_memory=True  # No multiprocessing for now for safety
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=0, pin_memory=True
    )
    
    return train_loader, val_loader, vocab
