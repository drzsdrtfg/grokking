import os
import requests
import json
import argparse
import tarfile
import zipfile
import gzip
import random
from tqdm import tqdm

# Set a default User-Agent to avoid blocking
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def download_file(url, destination, headers=None):
    """Download a file with progress bar and robust error handling"""
    if headers is None:
        headers = DEFAULT_HEADERS
    try:
        with requests.get(url, stream=True, headers=headers) as response:
            if response.status_code != 200:
                print(f"Failed to download {url} - Status code: {response.status_code}")
                return False
            
            # Allow text/plain content type for .txt files
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type and not url.endswith(('.html', '.txt')):
                print(f"Warning: Received HTML instead of expected file from {url}")
                return False
            
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            total_size = int(response.headers.get('content-length', 0))
            
            print(f"Downloading {url} to {destination} ({total_size/1024/1024:.1f} MB)")
            
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            
            with open(destination, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            
            progress_bar.close()
            
            if total_size > 0 and os.path.getsize(destination) < total_size:
                print(f"Warning: Download incomplete for {destination}")
                return False
                
            return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_file(file_path, extract_dir):
    """Extract compressed archives with error handling"""
    try:
        os.makedirs(extract_dir, exist_ok=True)
        
        if tarfile.is_tarfile(file_path):
            print(f"Extracting tar archive: {file_path}")
            with tarfile.open(file_path) as tar:
                tar.extractall(path=extract_dir)
            return True
            
        elif zipfile.is_zipfile(file_path):
            print(f"Extracting zip archive: {file_path}")
            with zipfile.ZipFile(file_path) as zip_ref:
                zip_ref.extractall(extract_dir)
            return True
            
        elif file_path.endswith('.gz'):
            output_path = os.path.join(extract_dir, os.path.basename(file_path)[:-3])
            print(f"Decompressing gzip: {file_path}")
            with gzip.open(file_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            return True
            
        print(f"No extraction needed for {file_path}")
        return True
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return False

def process_jsonl(jsonl_path, output_dir):
    """Process JSONL files into individual stories with 20% sampling"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        stories = []
        errors = 0
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing JSONL"):
                try:
                    data = json.loads(line)
                    # Handle different JSON structures and split by <endoftext>
                    text = ""
                    if 'text' in data:
                        text = data['text']
                    elif 'content' in data:
                        text = data['content']
                    elif 'story' in data:
                        text = data['story']
                    
                    # Split and add non-empty stories
                    if text:
                        stories.extend([s.strip() for s in text.split('<|endoftext|>') if s.strip()])
                        
                except json.JSONDecodeError:
                    errors += 1
        
        # Sample 20% of stories
        if stories:
            sample_size = max(1, int(len(stories) * 0.2))
            stories = random.sample(stories, sample_size)
        
        if errors > 0:
            print(f"Encountered {errors} parse errors")
        
        # Save individual stories
        for i, story in enumerate(tqdm(stories, desc="Saving stories")):
            with open(os.path.join(output_dir, f"story_{i+1:04d}.txt"), 'w', encoding='utf-8') as f:
                f.write(story)
        
        return True
    except Exception as e:
        print(f"Error processing {jsonl_path}: {e}")
        return False

def analyze_stories(processed_dir):
    """
    Analyze processed stories to find the largest story and total statistics
    Returns: (total_stories, total_chars, max_chars, largest_file)
    """
    max_chars = 0
    total_chars = 0
    total_stories = 0
    largest_file = None

    # Get all story files, excluding concatenated file
    story_files = [f for f in os.listdir(processed_dir) 
                  if f.startswith('story_') and f.endswith('.txt')]

    print(f"\nAnalyzing {len(story_files)} stories...")
    
    for filename in tqdm(story_files, desc="Processing stories"):
        file_path = os.path.join(processed_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                char_count = len(content)
                
                total_chars += char_count
                total_stories += 1
                
                if char_count > max_chars:
                    max_chars = char_count
                    largest_file = filename
                    
        except Exception as e:
            print(f"\nWarning: Could not process {filename} - {e}")

    return total_stories, total_chars, max_chars, largest_file

def try_multiple_sources():
    """Try multiple dataset sources with 20% sampling"""
    output_dir = "./tinystories_data"
    download_dir = os.path.join(output_dir, "downloads")
    extract_dir = os.path.join(output_dir, "extracted")
    processed_dir = os.path.join(output_dir, "processed")
    
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    sources = [
        {
            "name": "HuggingFace Dataset",
            "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
            "direct_text": True
        },
        {
            "name": "GitHub Archive",
            "url": "https://github.com/roneneldan/TinyStories/raw/main/TinyStories_all_data.zip",
            "is_archive": True
        }
    ]
    
    for source in sources:
        print(f"\nAttempting source: {source['name']}")
        fname = os.path.basename(source['url'])
        dl_path = os.path.join(download_dir, fname)
        
        # Download file
        if not os.path.exists(dl_path):
            if not download_file(source['url'], dl_path):
                continue
                
        # Process based on file type
        if source.get('direct_text'):
            # Direct text file processing
            with open(dl_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split and sample 20%
            stories = [s.strip() for s in content.split('<|endoftext|>') if s.strip()]
            if stories:
                sample_size = max(1, int(len(stories) * 0.2))
                stories = random.sample(stories, sample_size)
            
            print(f"Processing {len(stories)} stories (20% sample)")
            
            for i, story in enumerate(tqdm(stories)):
                with open(os.path.join(processed_dir, f"story_{i+1:04d}.txt"), 'w') as f:
                    f.write(story)
            
            return True
            
        elif source.get('is_archive'):
            # Archive processing
            if not extract_file(dl_path, extract_dir):
                continue
                
            # Process extracted files
            processed = False
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith('.jsonl'):
                        if process_jsonl(file_path, processed_dir):
                            processed = True
                    elif file.endswith('.txt'):
                        with open(file_path, 'r') as f:
                            stories = [s.strip() for s in f.read().split('<|endoftext|>') if s.strip()]
                        
                        # Sample 20%
                        if stories:
                            sample_size = max(1, int(len(stories) * 0.2))
                            stories = random.sample(stories, sample_size)
                        
                        for i, story in enumerate(stories):
                            with open(os.path.join(processed_dir, f"story_{i+1:04d}.txt"), 'w') as f_out:
                                f_out.write(story)
                        processed = True
            
            if processed:
                print("Successfully processed archive content")
                return True
    
    print("All sources failed, using fallback...")
    return False

def main():
    parser = argparse.ArgumentParser(description="TinyStories Dataset Downloader")
    parser.add_argument("--force", action="store_true", help="Force redownload files")
    args = parser.parse_args()
    
    if args.force:
        import shutil
        if os.path.exists("./tinystories_data"):
            shutil.rmtree("./tinystories_data")
    
    if try_multiple_sources():
        # Perform analysis after successful download
        processed_dir = "./tinystories_data/processed"
        total_stories, total_chars, max_chars, largest_file = analyze_stories(processed_dir)
        
        print("\nDataset statistics (20% sample):")
        print(f"Total stories: {total_stories:,}")
        print(f"Total characters: {total_chars:,}")
        print(f"Average story length: {total_chars/total_stories:,.0f} characters")
        print(f"Largest story: {largest_file} ({max_chars:,} characters)")
        print(f"\nDataset ready in {processed_dir}")
    else:
        print("\nGenerating synthetic stories...")
        # Add synthetic story generation here

if __name__ == "__main__":
    main()
    
