import openai
import os

OPENAI_API_KEY = ""
output_dir = '/home/captain/Desktop/NLP_FISAC/Backend/Hashtag_Generation'

def generate_keywords(topic, num_keywords = 20):
    prompt = f"Generate {num_keywords} keywords related to '{topic}' for research purposes."
    
    chat_client = openai.OpenAI(api_key=OPENAI_API_KEY)

    response = chat_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100
    )
    
    keywords = response.choices[0].message.content.strip()
    
    return keywords

def save_keywords_to_file(keywords, filename="keywords.txt"):
    with open(os.path.join(output_dir,filename), "w") as file:
        file.write(keywords)

def run(topic):
    keywords = generate_keywords(topic)
    save_keywords_to_file(keywords)
    print(f"keywords saved to keywords.txt")

if __name__ == "__main__":
    topic = input("Enter a topic: ")
    keywords = generate_keywords(topic)
    save_keywords_to_file(keywords)
    print(f"keywords saved to keywords.txt")
