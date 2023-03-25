import os
import json
import requests
import openai
import pandas as pd
import numpy as np
import tiktoken
from fastapi import FastAPI, Body, BackgroundTasks
from openai.embeddings_utils import get_embedding, cosine_similarity
from dotenv import load_dotenv
from bs4 import BeautifulSoup

datafile_path = "data/output.csv"

df = pd.read_csv(datafile_path)
df["ada_embedding"] = df["ada_embedding"].apply(eval).apply(np.array)

app = FastAPI()
load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY')
openai.organization = os.getenv('ORG_ID')
intercom_admin_id = os.getenv('intercom_admin_id')
intercom_assignee_id = os.getenv('intercom_assignee_id')
intercom_key = os.getenv('INTERCOM_KEY')
tag_id = "7930796"

chat_log_dir = 'chat_logs'
GET_DOCUMENTATION = "get_documentation"
PASS_TO_PERSON = "pass_to_person"
initial_system_instruction = "You are a friendly Hevo Support Assistant. You should always let user know that " \
                             "they need to type '0' to reach out to support representative."


# search through the reviews for a specific product
def search_documentation(query):
    product_embedding = get_embedding(
        query,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df["ada_embedding"].apply(lambda x: cosine_similarity(x, product_embedding))

    results = df.sort_values("similarity", ascending=False)
    response = ""
    c = 0
    for idx, result in results.iterrows():
        c = c + 1
        if result["n_tokens"] < 1000:
            response = result["combined"]
        else:
            response = result["url"]
        if c == 1:
            break
    return response


def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        print(f"Tokens: {num_tokens}")
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}. See 
        https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to 
        tokens.""")


def write_to_file(conversation_id: str, conversation: dict):
    with open(f"{chat_log_dir}/{conversation_id}.jsonl", "a") as outfile:
        outfile.write(json.dumps(conversation))
        outfile.write("\r\n")


def parse_user_message(user_message):
    parsed_user_message = BeautifulSoup(user_message, features="html.parser").get_text()
    return {
        "role": "user",
        "content": parsed_user_message
    }


def get_conversation(conversation_id: str):
    messages = []
    chat_file_path = f"{chat_log_dir}/{conversation_id}.jsonl"

    if os.path.exists(chat_file_path):
        with open(chat_file_path, 'r', encoding='utf-8') as f:
            messages = [json.loads(line) for line in f]
    else:
        messages.append({"role": "system", "content": initial_system_instruction})

    return messages


def save_conversation(conversation_id, messages):
    chat_file_path = f"{chat_log_dir}/{conversation_id}.jsonl"
    with open(chat_file_path, 'w', encoding='utf-8') as f:
        for item in messages:
            f.write(json.dumps(item, ensure_ascii=False) + '\r\n')


def post_to_intercom(conversation_id, message):
    request_body = {
        "message_type": "comment",
        "type": "admin",
        "admin_id": intercom_admin_id,
        "body": message.replace('\n', '<br>')
    }

    headers = {
        "Intercom-Version": "2.8",
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {intercom_key}"
    }

    intercom_reply_url = f"https://api.intercom.io/conversations/{conversation_id}/reply"
    requests.post(url=intercom_reply_url, data=json.dumps(request_body), headers=headers)


def assign_to_team(conversation_id):
    request_body = {
        "message_type": "assignment",
        "type": "team",
        "admin_id": intercom_admin_id,
        "assignee_id": intercom_assignee_id
    }

    headers = {
        "Intercom-Version": "2.8",
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {intercom_key}"
    }

    intercom_assign_url = f"https://api.intercom.io/conversations/{conversation_id}/parts"
    requests.post(url=intercom_assign_url, data=json.dumps(request_body), headers=headers)


def delete_tag(conversation_id):
    request_body = {
        "admin_id": intercom_admin_id
    }

    headers = {
        "Intercom-Version": "2.8",
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {intercom_key}"
    }

    intercom_delete_tag_url = f"https://api.intercom.io/conversations/{conversation_id}/tags/{tag_id}"
    requests.delete(url=intercom_delete_tag_url, data=json.dumps(request_body), headers=headers)


def get_gpt3_5_response(messages: list):
    # Todo: Optimize below hyper-parameters.
    print("Inside get_gpt3_5_response")
    print(messages)
    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    openai_chat_completion_url = "https://api.openai.com/v1/chat/completions"
    text_completion_response = requests.post(url=openai_chat_completion_url, data=json.dumps(request_body),
                                             headers=headers, timeout=30)

    print(f"text_completion_response: {text_completion_response}")
    text_completion = text_completion_response.json().get("choices")[0].get("message").get("content")
    return text_completion


def get_davinci_response(prompt: str):
    print(prompt)
    # Todo: Optimize below hyper-parameters.
    request_body = {
        "model": "text-davinci-003",
        "prompt": prompt,
        "max_tokens": 500,
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
    text_completion = \
        requests.post(url=openai_completion_url, data=json.dumps(request_body), headers=headers).json().get("choices")[
            0].get("text")
    return text_completion


def generate_response(conversation_id, user_message):
    messages = get_conversation(conversation_id=conversation_id)
    print("Got convo")

    if num_tokens_from_messages(messages=messages) > 2000:
        print("Summarize convo start")
        messages.append({
            "role": "system",
            "content": "Summarize the conversation in third person"
        })
        messages = [{
            "role": "system",
            "content": f"{initial_system_instruction}. "
                       f"Summary of conversation: {get_gpt3_5_response(messages=messages)}"
        }]
        print("Summarized convo success")

    messages.append({
        "role": "user",
        "content": user_message
    })

    messages.append(
        {
            "role": "system",
            "content": "Related Docs: "
                       + search_documentation(messages[-1].get("content")) +
                       "\nYou should always provide documentation link."
                       "\nYou should always let user know that "
                       "they need to type '0' in chat window to reach out to support representative."
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": get_gpt3_5_response(messages=messages)
        }
    )
    print("final GPT response")

    save_conversation(conversation_id=conversation_id, messages=messages)
    print("Save Convo")
    print("Post to intercom start")
    post_to_intercom(conversation_id=conversation_id, message=messages[-1].get("content"))
    print("Post to intercom success")


def pass_to_person(conversation_id):
    delete_tag(conversation_id=conversation_id)
    assign_to_team(conversation_id=conversation_id)


@app.post("/", status_code=200)
async def root(background_tasks: BackgroundTasks, data: dict = Body()):
    tags = data.get("data").get("item").get("tags").get("tags")

    for tag in tags:
        print(f"Request: {tag.get('id')}")
        if tag.get("id") == tag_id:
            if data.get("topic") == "conversation.user.created":
                user_message = data.get("data").get("item").get("source").get("body")
            else:
                user_message = data.get("data").get("item").get("conversation_parts").get("conversation_parts")[0].get(
                    "body")

            print(f"user message in gpt: {user_message}")
            conversation_id = data.get("data").get("item").get("id")
            user_message = BeautifulSoup(user_message, features="html.parser").get_text()
            if user_message == '0':
                print("Pass to person start")
                post_to_intercom(conversation_id=conversation_id,
                                 message="Passing your request to our support person. Tada!")
                pass_to_person(conversation_id=conversation_id)
                print("Pass to person success")
            else:
                background_tasks.add_task(generate_response, conversation_id, user_message)
            break
    return {"success": True}
