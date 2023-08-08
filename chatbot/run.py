import argparse

import os
import json
import requests
from salesgpt.agents import SalesGPT
from langchain.chat_models import ChatOpenAI


if __name__ == "__main__":

    # import your OpenAI key (put in your .env file)
    with open('.env','r') as f:
        env_file = f.readlines()
    envs_dict = {key.strip("'") :value.strip("\n") for key, value in [(i.split('=')) for i in env_file]}
    os.environ['OPENAI_API_KEY'] = envs_dict['OPENAI_API_KEY']

    # Initialize argparse
    parser = argparse.ArgumentParser(description='Description of your program')

    # Add arguments
    parser.add_argument('--config', type=str, help='Path to agent config file', default='')
    parser.add_argument('--verbose', type=bool, help='Verbosity', default=False)
    parser.add_argument('--max_num_turns', type=int, help='Maximum number of turns in the sales conversation', default=10)

    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    config_path = args.config
    verbose = args.verbose
    max_num_turns = args.max_num_turns


    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4") #model_name="gpt-4"
    agent = {}
    current_character = "danceadvocaat"
    available_characters = ["lunastar", "danceadvocaat"]
    storage_url = 'https://restaurantbotdb.blob.core.windows.net/profiles/'
    config_path = storage_url + current_character + '.txt'  # Set path to config if you have one
    USE_TOOLS=True

    def get_character(url):
        url = url
        response = requests.get(url)

        # Ensure the request was successful
        if response.status_code == 200:
            text = response.text
            return text
        else:
            print(f'Request failed with status {response.status_code}')


    def initialize_salesGPT_agent():
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
            print(f'Agent config {config}')
            newAgent = SalesGPT.from_llm(llm, verbose=True, **config)
        
        newAgent.seed_agent()
        return newAgent

    agent = initialize_salesGPT_agent()
    print('='*10)
    cnt = 0
    while cnt !=max_num_turns:
        cnt+=1
        if cnt==max_num_turns:
            print('Maximum number of turns reached - ending the conversation.')
            break
        agent.step()

        # end conversation 
        if '<END_OF_CALL>' in agent.conversation_history[-1]:
            print('Sales Agent determined it is time to end the conversation.')
            break
        human_input = input('Your response: ')
        agent.human_step(human_input)
        print('='*10)
