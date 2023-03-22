import os
import json
import requests
import openai
import pandas as pd
import numpy as np
from fastapi import FastAPI, Body, BackgroundTasks
from openai.embeddings_utils import get_embedding, cosine_similarity
from dotenv import load_dotenv
from bs4 import BeautifulSoup

datafile_path = "data/output.csv"

df = pd.read_csv(datafile_path)
df["ada_embedding"] = df.embedding.apply(eval).apply(np.array)

app = FastAPI()
load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY')
openai.organization = os.getenv('ORG_ID')
intercom_admin_id = os.getenv('intercom_admin_id')
intercom_key = os.getenv('INTERCOM_KEY')
chat_log_dir = 'chat_logs'


# search through the reviews for a specific product
def search_documentation(query, n=3, pprint=True):
    product_embedding = get_embedding(
        query,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df["ada_embedding"].apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results.str.cat(sep='\n')


def write_to_file(conversation_id: str, conversation: dict):
    with open(f"{chat_log_dir}/{conversation_id}.jsonl", "a") as outfile:
        outfile.write(json.dumps(conversation))
        outfile.write("\r\n")


def get_conversation(conversation_id: str, user_message: str):
    messages = [{"role": "system", "content": "You are a Hevo Data Support Assistant."}]
    chat_file_path = f"{chat_log_dir}/{conversation_id}.jsonl"

    if os.path.exists(chat_file_path):
        with open(chat_file_path) as reader:
            for line in reader:
                conversation = json.loads(line)
                messages.append({
                    "role": "user",
                    "content": conversation.get("prompt")
                })
                messages.append({
                    "role": "assistant",
                    "content": conversation.get("completion")
                })
    parsed_user_message = BeautifulSoup(user_message, features="html.parser").get_text()

    messages.append({
        "role": "system",
        "content": search_documentation(parsed_user_message)
    })

    # Todo: summarize prompt if it exceeds 1500 tokens.

    messages.append({
        "role": "user",
        "content": parsed_user_message
    })
    return messages


def post_to_intercom(message, conversation_id):
    request_body = {
        "message_type": "comment",
        "type": "admin",
        "admin_id": intercom_admin_id,
        "body": message
    }

    headers = {
        "Intercom-Version": "2.8",
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {intercom_key}"
    }

    intercom_reply_url = f"https://api.intercom.io/conversations/{conversation_id}/reply"
    requests.post(url=intercom_reply_url, data=json.dumps(request_body), headers=headers)


def get_gpt3_5_response(messages: list):
    # Todo: Optimize below hyper-parameters.
    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": 200,
        "temperature": 1
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    openai_chat_completion_url = "https://api.openai.com/v1/chat/completions"
    text_completion_response = requests.post(url=openai_chat_completion_url, data=json.dumps(request_body), headers=headers)
    text_completion = text_completion_response.json().get("choices")[0].get("message").get("content")
    return text_completion


def get_davinci_response(messages: list):

    prompt = "".join([f'{message["role"]}: {message["content"]} \n' for message in messages])
    prompt = prompt.replace('system:', '', 1) + "assistant: "
    # Todo: Optimize below hyper-parameters.
    request_body = {
        "model": "text-davinci-003",
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 1,
        "top_p": 1,
        "n": 1,
        "stream": False,
        "logprobs": None,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    openai_completion_url = "https://api.openai.com/v1/completions"
    text_completion = requests.post(url=openai_completion_url, data=json.dumps(request_body), headers=headers).json().get("choices")[0].get("text")
    return text_completion


def generate_response(data: dict = Body()):
    conversation_id = data.get("data").get("item").get("id")

    if data.get("topic") == "conversation.user.created":
        user_message = data.get("data").get("item").get("source").get("body")
    else:
        user_message = data.get("data").get("item").get("conversation_parts").get("conversation_parts")[0].get("body")

    messages = get_conversation(conversation_id=conversation_id, user_message=user_message)

    gpt_response = get_gpt3_5_response(messages=messages)

    write_to_file(conversation_id=conversation_id,
                  conversation={"prompt": messages[-1].get("content"), "completion": gpt_response})

    post_to_intercom(message=gpt_response, conversation_id=conversation_id)


@app.post("/", status_code=200)
async def root(background_tasks: BackgroundTasks, data: dict = Body()):
    background_tasks.add_task(generate_response, data=data)
    return {"success": True}
