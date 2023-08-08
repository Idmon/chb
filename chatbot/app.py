import os
import sys
import time
import json
import requests
import logging
import datetime
from functools import wraps
from flask import Flask, request, abort, send_from_directory
from dotenv import find_dotenv, load_dotenv
from streamlit_ui import streamlit_interface
from salesgpt.agents import SalesGPT
from langchain.chat_models import ChatOpenAI


import base64
from PIL import Image
import io

# Configure the logging level and format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Whatsapp API credentials
WHATSAPP_API_KEY = os.environ["WHATSAPP_API_KEY"]
WHATSAPP_BOT_ID = os.environ["WHATSAPP_BOT_ID"]

# Initialize SalesGPT
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4") #model_name="gpt-4"

available_characters = ["lunastar", "danceadvocaat", "giovanni", "roronoa"]
storage_url = 'https://restaurantbotdb.blob.core.windows.net/profiles/'

USE_TOOLS=True
agent_character = {}

def initialize_salesGPT_agent(character):
    newAgent=[]
    agent_character['character'] = character
    config_path = storage_url + character + '.txt'
    if config_path == '':
        print('No agent config specified, using a standard config')
        if USE_TOOLS:
            newAgent = SalesGPT.from_llm(llm, use_tools=True, 
                        product_catalog = None,
                        salesperson_name="Ted Lasso",
                        salesperson_role="Business Development Represantative",
                        verbose=True)
        else:
            newAgent = SalesGPT.from_llm(llm, verbose=True)
    else:
        character = get_character(config_path) 
        config = json.loads(character)
        newAgent = SalesGPT.from_llm(llm, verbose=True, **config)
    
    newAgent.seed_agent()
    
    return newAgent

def get_character(url):
    response = requests.get(url)

    # Ensure the request was successful
    if response.status_code == 200:
        text = response.text
        return text
    else:
        print(f'Request failed with status {response.status_code}')

# Initialize the Flask app
flask_app = Flask(__name__, static_folder='static')


@flask_app.route('/<path:filename>')  
def send_file(filename):  
    return send_from_directory(flask_app.static_folder, "image.png")

def require_whatsapp_verification(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not verify_whatsapp_request():
            abort(403)
        return f(*args, **kwargs)

    return decorated_function

def verify_whatsapp_request():
    incoming_msg = request.get_json()
    value = incoming_msg.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {})
    timestamp = value.get('messages', [{}])[0].get('timestamp', None)
    if not timestamp:
            return False

    try:
        timestamp = float(timestamp)
    except ValueError:
        return False

    current_time = datetime.datetime.now().timestamp()  # Current time in seconds since epoch
    time_difference = current_time - timestamp

    # check if timestamp is older than 30 sec, return False, if message is fresh return True
    if time_difference > 30:
        return False
    else:
        return True

def send_message(phone_number, message):
    url = f"https://graph.facebook.com/v17.0/{WHATSAPP_BOT_ID}/messages"  # Replace with your ID
    headers = {
        'Authorization': f"Bearer {WHATSAPP_API_KEY}",
        'Content-Type': 'application/json',
    }
    data = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {
            "body": message,
        },
    }

    requests.post(url, headers=headers, json=data)

def send_image(phone_number, image_url, caption=None):
    url = f"https://graph.facebook.com/v17.0/{WHATSAPP_BOT_ID}/messages"  # Replace with your bot ID
    headers = {
        'Authorization': f"Bearer {WHATSAPP_API_KEY}",
        'Content-Type': 'application/json',
    }
    data = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "image",
        "image": {
            "link": image_url,
        }
    }
    
    if caption:
        data["image"]["caption"] = caption

    response = requests.post(url, headers=headers, json=data)
    print(response.status_code)


def respond_agent(agent, text_msg):
    agent.human_step(text_msg)
    answer = agent.step()
    return answer

def save_base64_as_image(base64_str, save_path):
    # Decode the base64 string
    img_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(img_data))
    image.save(save_path)

def get_face(url):
    url = url
    response = requests.get(url)

    # Ensure the request was successful
    if response.status_code == 200:
        text = response.text
        return text
    else:
        print(f'Request failed with status {response.status_code}')


def generate_image(image_instructions):

    prompt, negPrompt, face =  agent_character['agent'].construct_prompt(image_instructions)
    print(prompt)

    url = "https://api.runpod.ai/v2/ldh8uvl63njv21/runsync"
    payload = {
        "input": {
            "api_name": "txt2img",
            "prompt": prompt,
            "negative_prompt": negPrompt,
            "width": 512,
            "height": 512,
            "guidance_scale": 3,
            "num_inference_steps": 38,
            "num_outputs": 1,
            "prompt_strength": 1,
            "scheduler": "DPM++ 2M Karras",
            "Face restoration": "CodeFormer"
        }
    } 

    args = []
    if face != '':
        print("FACE USED: " + face)
        base64_face = get_face(face)

        args=[
            base64_face, #0
            True #1 Enable ReActor
        ]

        payload["input"]["alwayson_scripts"] = {
            "reactor": {
                "args": args
            }
        }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        'Authorization': "Bearer U7UHFFKWS11VDMZEUFN4JQBC7XM5N1ZFOWHNZK7O",
    }

    response = requests.post(url, json=payload, headers=headers)

    # Assuming the response JSON is stored in a variable called response_json
    response_json = response.json()

    # Get the base64 image data of the first image
    base64_image_data = response_json['output']['images'][0]
    print(base64_image_data[:50])

    # Save path for the image
    save_path = 'static/image.png'

    # Call the function to save the base64 image data as an actual image
    save_base64_as_image(base64_image_data, save_path)


@flask_app.route('/webhook', methods=['GET', 'POST'])
@require_whatsapp_verification
def webhook():
    if request.method == 'GET':
        if request.args.get('hub.verify_token') == 'chatbot':
            return request.args.get('hub.challenge')
        return 'Invalid verify token', 403
    elif request.method == 'POST':
        # here you get the incoming message from the WhatsApp Business API
        incoming_msg = request.get_json()
        print(incoming_msg)
        
        # extract the necessary information you need from the message 
        value = incoming_msg.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {})
        phone_number = value.get('messages', [{}])[0].get('from', None)

        if phone_number is not None:
            text_msg = value.get('messages', [{}])[0].get('text', {}).get('body', '')

            if text_msg in available_characters:
                agent_character['agent'] = initialize_salesGPT_agent(text_msg)
            else:
                if 'agent' not in agent_character:
                    agent_character['agent'] = initialize_salesGPT_agent(available_characters[0])  # Using the first available character as default.

                # Pass the message to chatbot for processing
                answer = respond_agent(agent_character['agent'], text_msg)

                if '<PROCESS_IMAGE>' in answer:
                    answer = answer.replace("<PROCESS_IMAGE>", "")
                    send_message(phone_number, answer)
                    image_prompt = agent_character['agent'].create_image_prompt()
                    generate_image(image_prompt)
                    agent_character['agent'].system_step("<IMAGE_READY>")
                    answer = agent_character['agent'].step()
                    answer = answer.replace("<IMAGE>", "")
                    send_image(phone_number, "https://gf-chatbot.azurewebsites.net/image.png")
                    send_message(phone_number, answer)
                else:
                    send_message(phone_number, answer)

        return 'OK', 200

# Run the Flask app
if __name__ == "__main__":
    agent_character['agent'] = initialize_salesGPT_agent(available_characters[0])

    environment = os.getenv('ENVIRONMENT', 'server')  # Defaults to 'server' if the ENVIRONMENT variable is not set
    if environment == 'streamlit':
        logging.info("Streamlit started")
        streamlit_interface(agent_character['agent'])
    else:
        logging.info("Flask app started")
        flask_app.run(host="0.0.0.0", port=8000)
