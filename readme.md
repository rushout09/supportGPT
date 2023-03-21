# Add GPT to your intercom Chat.

### Known limitations:

1. Signed webhooks from intercom are not verified yet.
2. Yet to write steps to deploy in production.

### Steps to set up:
1. Clone the repo using: 
```
 git clone git@github.com:rushout09/supportGPT.git
```
2. Add a .env file with following details: 
```
OPENAI_KEY=
ORG_ID=
INTERCOM_KEY=
intercom_admin_id=
```
3. Create a virtual env and Install the requirements file:
```
python3 -n venv venv
pip install -r requirements.txt
```
4. Install cloudflare tunnel to proxy your localhost (Optional)
```
brew install cloudflared
cloudflared tunnel --url http://localhost:8000
```
5. Follow this guide to setup webhooks on intercom: https://developers.intercom.com/building-apps/docs/setting-up-webhooks
6. Run the following command to start the server:
```
uvicorn main:app --reload
```