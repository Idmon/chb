import json
import requests
import datetime

def send_test_message(message):
    url = 'http://127.0.0.1:8000/webhook'  # replace with the actual URL of your webhook
    headers = {
        'Content-Type': 'application/json',
    }
    current_timestamp = int(datetime.datetime.now().timestamp())
    data = {
        'entry': [{
            'changes': [{
                'value': {
                    'messages': [{
                        'timestamp': str(current_timestamp),
                        'from': '31631908540',  # replace with the actual phone number
                        'text': {
                            'body': message,
                        },
                    }],
                },
            }],
        }],
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        print('OK')
    else:
        print(f'Failed to send message. Status code: {response.status_code}')


if __name__ == '__main__':
    while True:
        message = input("Enter your message (type 'quit' to exit): ")
        if message.lower() == 'quit':
            break
        send_test_message(message)
