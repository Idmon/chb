import requests
import os
import base64
from PIL import Image
import io
import json

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


def generate_image():

    base64_face = get_face("https://restaurantbotdb.blob.core.windows.net/profiles/face.txt")

    args=[
        base64_face, #0
        True #1 Enable ReActor
    ]


    url = "https://api.runpod.ai/v2/shhdv5w58hhanm/runsync"
    payload = {
        "input": {
            "api_name": "txt2img",
            "prompt": "frontdoggy, 1girl, 1boy, breasts, nipples, hetero, sex, sex from behind, all fours, doggystyle <lora:doggystylefront:1>",
            "negative_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",
            "width": 512,
            "height": 512,
            "guidance_scale": 3,
            "num_inference_steps": 38,
            "num_outputs": 1,
            "prompt_strength": 1,
            "scheduler": "DPM++ 2M Karras",
            "Face restoration": "CodeFormer",
            "enable_hr": False,
            "Denoising strength": 0.3,
            "Hires upscale": 1.5,
            "Hires upscaler": "superscale",
            "Hires steps": 30,
            "alwayson_scripts": {
                "reactor":{
                    "args":args
                    }
                }
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

    #print(response_json)

    # Get the base64 image data of the first image
    base64_image_data = response_json['output']['images'][0]
    print(base64_image_data[:50])

    # Save path for the image
    save_path = 'image.png'

    # Call the function to save the base64 image data as an actual image
    save_base64_as_image(base64_image_data, save_path)

generate_image()