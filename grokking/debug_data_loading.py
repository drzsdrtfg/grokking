import os
import torch
import glob
import random

def debug_tinystories_loading(data_path, seq_len=64, min_seq_len=20):
    """
    Debug the TinyStories loading process with extensive logging
    """
    print(f"Data path: {data_path}")
    
    # Check if directory exists
    if not os.path.exists(data_path):
        print(f"ERROR: Data path {data_path} does not exist")
        return False
    
    # List all files in the directory
    all_files = glob.glob(os.path.join(data_path, "*.txt"))
    print(f"Found {len(all_files)} text files in {data_path}")
    
    if len(all_files) == 0:
        print("ERROR: No text files found in the data path")
        return False
    
    # Check a few random files to see their content
    if len(all_files) > 0:
        sample_files = random.sample(all_files, min(3, len(all_files)))
        print(f"Sampling {len(sample_files)} files to check content:")
        
        for file_path in sample_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"\nFile: {os.path.basename(file_path)}")
                print(f"Length: {len(content)} characters")
                print(f"First 100 chars: {content[:100]}...")
                
                if len(content) < min_seq_len:
                    print(f"WARNING: Content is very short ({len(content)} chars)")
            except Exception as e:
                print(f"ERROR reading file {file_path}: {e}")
    
    # Try to determine if we have any valid data for training
    valid_files = []
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if len(content) >= min_seq_len:
                valid_files.append(file_path)
        except:
            continue
    
    print(f"\nFound {len(valid_files)} files with sufficient content for training")
    
    if len(valid_files) == 0:
        print("ERROR: No valid training data found")
        return False
    
    # If we have valid files, try creating a small sample dataset
    print("\nTrying to create a sample dataset...")
    
    # Build a simple character vocabulary
    try:
        all_chars = set()
        for file_path in valid_files[:5]:  # Just use a few files for testing
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                all_chars.update(content)
        
        char_vocab = {char: i for i, char in enumerate(sorted(list(all_chars)))}
        # Add special tokens
        char_vocab['<bos>'] = len(char_vocab)
        char_vocab['<eos>'] = len(char_vocab)
        char_vocab['<pad>'] = len(char_vocab)
        
        print(f"Created character vocabulary with {len(char_vocab)} tokens")
        
        # Create a sample sequence
        sample_file = valid_files[0]
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Tokenize a sample sequence
        tokens = [char_vocab.get(c, char_vocab['<pad>']) for c in content[:seq_len*2]]
        print(f"Tokenized {len(tokens)} characters from sample file")
        
        # Create input and target sequences
        if len(tokens) > seq_len + 1:
            input_seq = tokens[:seq_len]
            target_seq = tokens[1:seq_len+1]
            
            # Prepend BOS token
            input_seq = [char_vocab['<bos>']] + input_seq[:-1]
            
            print(f"Created sample input sequence of length {len(input_seq)}")
            print(f"Created sample target sequence of length {len(target_seq)}")
            
            return True
        else:
            print("ERROR: Sample sequence is too short")
            return False
        
    except Exception as e:
        print(f"ERROR during dataset creation: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python debug_data_loading.py <data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    success = debug_tinystories_loading(data_path)
    
    if success:
        print("\nDEBUG SUCCESS: Data loading and processing works correctly")
    else:
        print("\nDEBUG FAILED: Issues found with data loading")