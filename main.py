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
admin_id = os.getenv('intercom_admin_id')
intercom_key = os.getenv('INTERCOM_KEY')
chat_log_dir = 'chat_logs'


def respond_to_user(data: dict = Body()):
    user_topic = data.get("topic")
    conversation_id = data.get("data").get("item").get("id")
    if user_topic == "conversation.user.created":
        user_message = data.get("data").get("item").get("source").get("body")
        prompt = "You are a HevoData Support Bot."
    else:
        user_message = data.get("data").get("item").get("conversation_parts").get("conversation_parts")[0]. \
            get("body")
        with open(f"{chat_log_dir}/{conversation_id}.json") as reader:
            conversation = json.loads(reader.read())
        prompt = conversation.get("prompt") + conversation.get("completion")
        # Todo: summarize prompt if it exceeds 1500 tokens.

    soup = BeautifulSoup(user_message, features="html.parser")
    prompt = prompt + "\n User: " + soup.get_text() + "\n Bot: "
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
    url = "https://api.openai.com/v1/completions"
    text_completion = requests.post(url=url, data=json.dumps(request_body), headers=headers).json().get("choices")[0]. \
        get("text")

    conversation = {
        "prompt": prompt,
        "completion": text_completion
    }

    headers = {
        "Intercom-Version": "2.8",
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {intercom_key}"
    }

    request_body = {
        "message_type": "comment",
        "type": "admin",
        "admin_id": admin_id,
        "body": text_completion
    }

    intercom_reply_url = f"https://api.intercom.io/conversations/{conversation_id}/reply"
    requests.post(url=intercom_reply_url, data=json.dumps(request_body), headers=headers)

    with open(f"{chat_log_dir}/{conversation_id}.json", "w") as outfile:
        outfile.write(json.dumps(conversation))


@app.post("/", status_code=200)
async def root(background_tasks: BackgroundTasks, data: dict = Body()):
    background_tasks.add_task(respond_to_user, data=data)
    return {"success": True}
