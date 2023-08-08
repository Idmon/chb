import requests

url = "https://api.runpod.ai/v2/whxsmjssnq3qlu/status/sync-607211d5-b7b9-40e2-9e1a-537bc44b46f6"

headers = {
    "accept": "application/json",
    'Authorization': f"Bearer U7UHFFKWS11VDMZEUFN4JQBC7XM5N1ZFOWHNZK7O",
}

response = requests.post(url, headers=headers)

print(response.text)