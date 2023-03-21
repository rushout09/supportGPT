import os
import json
import requests
import openai
from fastapi import FastAPI, Body, BackgroundTasks
from dotenv import load_dotenv
from bs4 import BeautifulSoup

app = FastAPI()
load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY')
openai.organization = os.getenv('ORG_ID')
intercom_admin_id = os.getenv('intercom_admin_id')
intercom_key = os.getenv('INTERCOM_KEY')
chat_log_dir = 'chat_logs'


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


def gpt3_5_response(data: dict = Body()):
    user_topic = data.get("topic")
    conversation_id = data.get("data").get("item").get("id")
    messages = [{"role": "system", "content": "You are a Hevo Data Support Assistant."}]

    if user_topic == "conversation.user.created":
        user_message = data.get("data").get("item").get("source").get("body")
    else:
        user_message = data.get("data").get("item").get("conversation_parts").get("conversation_parts")[0]. \
            get("body")
        with open(f"{chat_log_dir}/{conversation_id}.jsonl") as reader:
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
        # Todo: summarize prompt if it exceeds 1500 tokens.

    parsed_user_message = BeautifulSoup(user_message, features="html.parser").get_text()
    messages.append({
        "role": "user",
        "content": parsed_user_message
    })
    # Todo: Optimize below hyper-parameters.
    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": messages
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    print(json.dumps(request_body))
    openai_chat_completion_url = "https://api.openai.com/v1/chat/completions"
    text_completion_response = requests.post(url=openai_chat_completion_url, data=json.dumps(request_body), headers=headers)
    print(text_completion_response.text)
    text_completion = text_completion_response.json().get("choices")[0].get("message").get("content")

    conversation = {
        "prompt": parsed_user_message,
        "completion": text_completion
    }

    with open(f"{chat_log_dir}/{conversation_id}.jsonl", "a") as outfile:
        outfile.write(json.dumps(conversation))
        outfile.write("\r\n")

    post_to_intercom(message=text_completion, conversation_id=conversation_id)


def davinci_response(data: dict = Body()):
    user_topic = data.get("topic")
    conversation_id = data.get("data").get("item").get("id")

    prompt = "\n system: " + "You are a HevoData Support Assistant."
    if user_topic == "conversation.user.created":
        user_message = data.get("data").get("item").get("source").get("body")
    else:
        user_message = data.get("data").get("item").get("conversation_parts").get("conversation_parts")[0]. \
            get("body")
        prompt = ""
        with open(f"{chat_log_dir}/{conversation_id}.jsonl") as reader:
            for line in reader:
                conversation = json.loads(line)
                prompt = prompt + "\n user: " + conversation.get("prompt") + "\n assistant: " + conversation.get("completion")
        # Todo: summarize prompt if it exceeds 1500 tokens.

    parsed_user_message = BeautifulSoup(user_message, features="html.parser").get_text()
    prompt = prompt + "\n user: " + parsed_user_message + "\n assistant: "
    # Todo: Optimize below hyper-parameters.
    request_body = {
        "model": "text-davinci-003",
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "stream": False,
        "logprobs": None,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    print(json.dumps(request_body))
    openai_completion_url = "https://api.openai.com/v1/completions"
    text_completion = requests.post(url=openai_completion_url, data=json.dumps(request_body), headers=headers).json().get("choices")[0].get("text")

    conversation = {
        "prompt": parsed_user_message,
        "completion": text_completion
    }

    with open(f"{chat_log_dir}/{conversation_id}.jsonl", "a") as outfile:
        outfile.write(json.dumps(conversation))
        outfile.write("\r\n")

    post_to_intercom(message=text_completion, conversation_id=conversation_id)


@app.post("/", status_code=200)
async def root(background_tasks: BackgroundTasks, data: dict = Body()):
    background_tasks.add_task(gpt3_5_response, data=data)
    return {"success": True}
