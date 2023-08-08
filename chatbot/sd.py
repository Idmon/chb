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


    url = "https://api.runpod.ai/v2/ldh8uvl63njv21/runsync"
    payload = {
        "input": {
            "api_name": "txt2img",
            "prompt": f"best quality, beautiful, 23-year-old Lebanese girl, long brown hair, ((large breast)), sexy look, (titfuck), (pink bra:1.2), (Egypt at night, Ocean view:1.2) <lora:titfuck:0.8>",
            "negative_prompt": "bad-hands-5, bad-artist, watermark, twin, second person",
            "width": 700,
            "height": 700,
            "guidance_scale": 7,
            "num_inference_steps": 25,
            "num_outputs": 1,
            "prompt_strength": 1,
            "scheduler": "DPM++ 2M Karras",
            "Face restoration": "CodeFormer",
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