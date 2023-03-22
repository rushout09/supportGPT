import os
import pandas as pd
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv
import tiktoken
from time import sleep
load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY')
openai.organization = os.getenv('ORG_ID')

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191


def get_embedding(combined_list, model="text-embedding-ada-002"):
    embedding_list = []
    for combined in combined_list:
        while True:
            try:
                embedding_list.append(openai.Embedding.create(input=[combined], model=model)['data'][0]['embedding'])
                print(combined)
                print(len(embedding_list))
                break
            except openai.error.RateLimitError as e:
                # If the API call fails, wait and retry after a delay
                print("API error:", e)
                print("Retrying in 10 seconds...")
                sleep(5)
    return embedding_list


# Define function to extract title, content, and url from a single MD file
def process_md_file(md_file):
    with open(md_file, 'r') as f:
        # Read the file contents
        html = f.read()
        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        # Extract the title and content
        title = soup.title.string.strip() if soup.title else ''
        title = title.replace('- Hevo Data', '')
        title = title.strip()
        content = soup.body.get_text().strip().replace('\n', '') if soup.body else ''
        content = content.replace('Share', '', 1)
        content = content.replace(title, '', 1)
        content = content[:content.find("Last updated on ")]
        content = content.strip()

        url = f"https://docs.hevodata.com/{md_file[len(docs_dir)+1:-10]}"
    return {"title": title, "content": content, "url": url}


# Define function to process all MD files in a directory and store the results in a DataFrame
def process_md_dir(md_dir):
    md_files = []
    for root, dirs, files in os.walk(md_dir, topdown=True):
        if "release-notes" not in root:
            md_files.extend([os.path.join(root, f) for f in files if f.endswith('.html')])

    data = [process_md_file(f) for f in md_files]
    data_df = pd.DataFrame(data)

    data_df = data_df[3:]

    data_df["combined"] = (
            "Title: " + data_df["title"].str.strip() + "; Content: " + data_df["content"].str.strip() + "; Url: " +
            data_df["url"].str.strip()
    )

    encoding = tiktoken.get_encoding(embedding_encoding)
    data_df["n_tokens"] = data_df["combined"].apply(lambda x: len(encoding.encode(x)))
    data_df = data_df[data_df.n_tokens <= max_tokens]
    data_df.head()

    data_df['ada_embedding'] = get_embedding(data_df["combined"].tolist(), model=embedding_model)
    data_df.head()
    data_df.to_csv("output.csv")


# Process all MD files in the specified directory and save as CSV
docs_dir = "/Users/rushabh.agarwal/mini-docs"
process_md_dir(md_dir=docs_dir)



