import os
import sys
import time
import json
import requests
import logging
import datetime
from functools import wraps
from flask import Flask, request, abort, send_from_directory, jsonify, Response
from dotenv import find_dotenv, load_dotenv
from streamlit_ui import streamlit_interface
from salesgpt.agents import SalesGPT
from salesgpt.logger import in_memory_handler
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

class SuppressLogOutputFilter(logging.Filter):
    def filter(self, record):
        return '/log-output' not in record.getMessage()

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Whatsapp API credentials
WHATSAPP_API_KEY = os.environ["WHATSAPP_API_KEY"]
WHATSAPP_BOT_ID = os.environ["WHATSAPP_BOT_ID"]

# Initialize SalesGPT
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4") #model_name="gpt-4"

available_characters = ["lunastar", "luna", "danceadvocaat", "giovanni", "roronoa", "vastgoedheer"]
storage_url = 'https://restaurantbotdb.blob.core.windows.net/profiles/'

USE_TOOLS=True
agent_character = {}
verbose = False

def initialize_salesGPT_agent(character):
    newAgent=[]
    agent_character['character'] = character
    config_path = storage_url + character + '.txt'
    if config_path == '':
        logger.info('No agent config specified, using a standard config')
        if USE_TOOLS:
            newAgent = SalesGPT.from_llm(llm, use_tools=True, 
                        product_catalog = None,
                        salesperson_name="Ted Lasso",
                        salesperson_role="Business Development Represantative",
                        verbose=verbose)
        else:
            newAgent = SalesGPT.from_llm(llm, verbose=verbose)
    else:
        character = get_character(config_path) 
        config = json.loads(character)
        newAgent = SalesGPT.from_llm(llm, verbose=verbose, **config)
    
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
logger = logging.getLogger("App")
# log = logging.getLogger('werkzeug')
# log.addFilter(SuppressLogOutputFilter())
flask_app = Flask(__name__, static_folder='static')


# @flask_app.errorhandler(Exception)
# def handle_exception(e):
#     # Log the error
#     logger.error(f"Unhandled exception: {str(e)}", exc_info=True)

#     # You can either return a string with the error here or render an error template, or jsonify the error
#     return jsonify(error=str(e)), 500  # or render_template('error_page.html'), or just str(e), depending on your use-case


# @flask_app.route('/log-output', methods=['GET'])
# def get_logs():
#     logs = in_memory_handler.get_logs()
#     return jsonify(logs)


# @flask_app.route('/logs', methods=['GET'])
# def log_viewer():
#     return '''
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <meta http-equiv="X-UA-Compatible" content="IE=edge">
#         <meta name="viewport" content="width=device-width, initial-scale=1.0">
#         <title>Live Logs</title>
#     </head>
#     <body>
#         <div id="log-content"></div>

#         <script>
#             function fetchLogs() {
#                 fetch('/log-output')
#                     .then(response => response.json())
#                     .then(data => {
#                         const logs = data.map(log => `<pre style='margin:0;'>${log}</pre>`).join('\\n');
#                         document.getElementById('log-content').innerHTML = logs;
#                     })
#                     .catch(error => console.error('Error fetching logs:', error));
#             }

#             setInterval(fetchLogs, 5000);  // Refresh every 5 seconds
#             fetchLogs();  // Initial fetch
#         </script>
#     </body>
#     </html>
#     '''



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
    logger.info("Image Instructions: " + str(image_instructions))

    base64_face = get_face(face)
    args=[
        base64_face, #0
        True, #1 Enable ReActor
        '0', #2 Comma separated face number(s) from swap-source image
        '0', #3 Comma separated face number(s) for target image (result)
        '/stable-diffusion-webui/models/roop/inswapper_128.onnx', #4 model path
        'CodeFormer', #4 Restore Face: None; CodeFormer; GFPGAN
        1, #5 Restore visibility value
        True, #7 Restore face -> Upscale
    ]

    url = "https://api.runpod.ai/v2/shhdv5w58hhanm/runsync"
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
            "enable_hr": True,
            "denoising_strength": 0.3,
            "hires_upscale": 1.25,
            "hires_upscaler": "superscale",
            "hires_steps": 30,
            "alwayson_scripts": {
                "reactor":{
                    "args":args
                    }
                }
        }
    } 

    # args = []
    # if face != '':
    #     print("FACE USED: " + face)
    #     base64_face = get_face(face)

    #     args=[
    #         base64_face, #0
    #         True #1 Enable ReActor
    #     ]

    #     payload["input"]["alwayson_scripts"] = {
    #         "reactor": {
    #             "args": args
    #         }
    #     }

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
    logger.info("Image downloaded" + base64_image_data[:50])

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
        
        # extract the necessary information you need from the message 
        value = incoming_msg.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {})
        phone_number = value.get('messages', [{}])[0].get('from', None)

        if phone_number is not None:
            text_msg = value.get('messages', [{}])[0].get('text', {}).get('body', '')
            logger.info("User: " + text_msg)

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
                    send_image(phone_number, "https://gf-bot.azurewebsites.net/image.png")
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
