import os
import random

# A collection of simple TinyStories-like stories for testing
SAMPLE_STORIES = [
    """Once upon a time, there was a little red bird. The bird loved to sing. Every morning, the bird sang a happy song. All the animals in the forest liked to hear the little bird sing. One day, the bird could not sing. It was very sad. A kind fox asked, "Why are you sad, little bird?" The bird said, "I lost my song!" The fox helped the bird look for its song. They looked high and low. Then, the bird drank some water. Suddenly, it could sing again! The little bird was very happy. It sang a special song for the fox. They became good friends.""",
    
    """Tim had a small toy car. The car was blue and fast. Tim played with his car every day. One day, Tim could not find his car. He looked under his bed. He looked in his toy box. He looked on his desk. The car was not there. Tim was sad. Then, his dog Max came into the room. Max had the blue car in his mouth! Tim laughed and said, "Max, that's my car!" Max gave the car back to Tim. Tim gave Max a big hug. Then they played together all day.""",
    
    """Lily wanted to plant a flower. She got a small pot and some soil. She put the soil in the pot. Then, she made a little hole in the soil. Lily put a seed in the hole and covered it with soil. Every day, Lily gave the pot some water. She put the pot near the window so it could get sun. Lily waited and waited. One week later, she saw a tiny green leaf! Lily was so happy. Her plant was growing! After many days, a beautiful purple flower bloomed. Lily showed everyone her pretty flower.""",
    
    """Sam and Mia went to the park. It was a sunny day. They saw ducks in the pond. Sam had some bread. He broke the bread into small pieces. Sam and Mia fed the ducks. The ducks quacked and swam close to get the bread. Then, Sam and Mia played on the swings. They went up high. They could see the whole park! After that, they had ice cream. Sam got chocolate. Mia got strawberry. It was a very fun day at the park.""",
    
    """Ben had a big box. He put the box in his room. Ben climbed inside the box. He pretended the box was a boat. Ben sailed on the ocean. He saw fish and whales. Then, Ben pretended the box was a car. He drove on a road. He went very fast! Next, Ben said the box was a rocket ship. He flew to the moon and stars. Ben's mom called, "Time for dinner!" Ben came out of his box. "I'll be back to play more," he told his box. Ben's box was the best toy ever.""",
    
    """Zoe found a magic stone. The stone was small and blue. When Zoe held the stone, she could talk to animals. She talked to her cat, Whiskers. Whiskers said, "I love when you pet my ears." Zoe talked to a bird outside. The bird said, "I'm building a nest in your tree." Zoe talked to a squirrel. The squirrel said, "I've hidden nuts all over your yard!" Zoe had fun talking to all the animals. But then, she dropped the stone. It fell into a puddle. After that, Zoe couldn't talk to animals anymore. But sometimes, she still thinks Whiskers answers her.""",
    
    """Jack and Emma built a snow fort. The snow was deep and white. They made a big wall. They made a door to go inside. Jack found sticks to put on top. Emma put her red scarf on the door. Their fort was the best! They sat inside and drank hot chocolate. It was warm and cozy in their fort. Soon, other kids came to see. "Can we play in your fort?" they asked. Jack and Emma said, "Yes!" All the children played in the snow fort until the sun went down. It was a perfect snow day.""",
    
    """Penny got a new puppy. The puppy was small with brown spots. Penny named him Spot. Spot liked to chew on shoes. He liked to run and jump. But Spot didn't know how to sit or stay. Penny wanted to teach Spot. She held a treat in her hand. "Sit, Spot!" said Penny. Spot looked at her. Then he sat down! Penny gave him the treat. "Good boy!" she said. Every day, Penny taught Spot something new. Soon, Spot could sit, stay, and come when called. Penny was proud of her smart puppy."""
]

def create_sample_tinystories(output_dir, num_stories=50):
    """
    Create a sample TinyStories dataset for testing
    """
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Generate multiple stories by combining and slightly modifying sample stories
    all_stories = []
    for i in range(num_stories):
        # Select 1-2 stories to combine/use
        story_templates = random.sample(SAMPLE_STORIES, k=min(2, len(SAMPLE_STORIES)))
        
        if len(story_templates) == 1 or random.random() < 0.7:
            # Use a single story with minor modifications
            story = story_templates[0]
            
            # Simple modifications
            if random.random() < 0.3:
                # Change character names
                name_replacements = {
                    "Tim": random.choice(["Tom", "Ben", "Max", "Leo"]),
                    "Lily": random.choice(["Lucy", "Zoe", "Mia", "Anna"]),
                    "Sam": random.choice(["Jake", "Alex", "Will", "Ryan"]),
                    "Mia": random.choice(["Emma", "Sara", "Lily", "Kate"]),
                    "Ben": random.choice(["Tim", "Sam", "Jack", "Noah"]),
                    "Zoe": random.choice(["Lily", "Emma", "Ava", "Mia"]),
                    "Jack": random.choice(["Sam", "Ben", "Tom", "Alex"]),
                    "Emma": random.choice(["Zoe", "Mia", "Lily", "Sara"]),
                    "Penny": random.choice(["Lucy", "Zoe", "Emma", "Ava"])
                }
                
                for old_name, new_name in name_replacements.items():
                    if old_name in story:
                        story = story.replace(old_name, new_name)
            
            if random.random() < 0.3:
                # Change colors
                color_replacements = {
                    "red": random.choice(["blue", "green", "yellow", "purple"]),
                    "blue": random.choice(["red", "green", "yellow", "purple"]),
                    "green": random.choice(["blue", "red", "yellow", "purple"]),
                    "purple": random.choice(["blue", "green", "red", "yellow"])
                }
                
                for old_color, new_color in color_replacements.items():
                    if old_color in story:
                        story = story.replace(old_color, new_color)
        else:
            # Combine two stories
            story = story_templates[0] + "\n\n" + story_templates[1]
        
        all_stories.append(story)
    
    # Save individual stories
    for i, story in enumerate(all_stories):
        with open(os.path.join(processed_dir, f"story_{i+1}.txt"), 'w', encoding='utf-8') as f:
            f.write(story)
    
    # Save combined file
    with open(os.path.join(processed_dir, "all_stories.txt"), 'w', encoding='utf-8') as f:
        f.write("\n\n".join(all_stories))
    
    print(f"Created {len(all_stories)} sample stories in {processed_dir}")
    print(f"You can now use it for training with: --mode tinystories --data_path {processed_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create sample TinyStories for testing")
    parser.add_argument("--output_dir", type=str, default="./tinystories_data",
                        help="Directory to save the generated stories")
    parser.add_argument("--num_stories", type=int, default=50,
                        help="Number of stories to generate")
    
    args = parser.parse_args()
    create_sample_tinystories(args.output_dir, args.num_stories)