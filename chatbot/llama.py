import requests
import os
import base64
from PIL import Image
import io
import json

def generate_prompt():

    url = "https://api.runpod.ai/v2/m5qshknu0v6gq5/runsync"
    payload = {
        "input": {
            "prompt": "hi there",
            "max_new_tokens": 500,
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "batch_size": 8,
            "stop": ["</s>"]
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
    print(response_json)

generate_prompt()