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
            "prompt": "best quality, beautiful, 23-year-old Lebanese supermodel, full body, long brown hair,  wearing (Olympian dress), tanding in a modern minimalist (mount lympus), perfect face, perfect lighting, intricate, incredible detail, masterpiece, professional photoshoot",
            "negative_prompt": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",
            "width": 512,
            "height": 512,
            "guidance_scale": 3,
            "num_inference_steps": 38,
            "num_outputs": 1,
            "prompt_strength": 1,
            "scheduler": "DPM++ 2M Karras",
            # "Face restoration": "CodeFormer",
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