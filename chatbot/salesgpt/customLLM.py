import os
import re
from dotenv import find_dotenv, load_dotenv
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests
from pydantic import BaseModel, Field


load_dotenv(find_dotenv())

HOST = os.environ["OB_API_HOST"]

class HistoryModel(BaseModel):
    internal: List[List[str]] = []
    visible: List[List[str]] = []

class customChatLLM(LLM):

    history: HistoryModel = HistoryModel(internal=[], visible=[])

    @staticmethod
    def transform_to_history_model(conversation_history: List[str]) -> HistoryModel:

        def clean_visible_item(item: str) -> str:
            # Remove any string that starts with <, contains text, and ends with >
            item = re.sub(r'<[^>]+>', '', item)
            # Replace specific message to an empty string
            if item == "SYSTEM: <IMAGE_READY>":
                return ""
            return item

        transformed_data = {
            'internal': [[item] for item in conversation_history],
            'visible': [[clean_visible_item(item)] for item in conversation_history]
        }

        return HistoryModel(**transformed_data)

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, conversation_history: List[str], stop: Optional[List[str]] = None) -> str:

        history = customChatLLM.transform_to_history_model(conversation_history)
        
        URI = f'https://{HOST}/api/v1/chat'

        request = {
            'user_input': "prompt here",
            'max_new_tokens': 300,
            'auto_max_new_tokens': False,
            'mode': 'chat',  # Valid options: 'chat', 'chat-instruct', 'instruct'
            'character': 'LUNAv47',
            'instruction_template': 'Orca-send-image2',  # Will get autodetected if unset
            # 'your_name': 'Idmon',
            # 'name1': 'Idmon', # Optional
            # 'name2': 'Hakeem', # Optional
            # 'context': 'character context', # Optional
            # 'greeting': 'greeting', # Optional
            # 'name1_instruct': 'You', # Optional
            # 'name2_instruct': 'Assistant', # Optional
            # 'context_instruct': 'context_instruct', # Optional
            # 'turn_template': 'turn_template', # Optional
            'regenerate': False,
            '_continue': False,
            'stop_at_newline': False,
            'chat_generation_attempts': 1,
            'chat_instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',

            # Generation params. If 'preset' is set to different than 'None', the values
            # in presets/preset-name.yaml are used instead of the individual numbers.
            'preset': 'None',
            'temperature': 0.7,
            'top_p': 0.1,
            'top_k': 40,
            'typical_p': 1,
            'epsilon_cutoff': 0,  # In units of 1e-4
            'eta_cutoff': 0,  # In units of 1e-4
            'tfs': 1,
            'top_a': 0,
            'repetition_penalty': 1.05,
            'repetition_penalty_range': 0,
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'mirostat_mode': 0,
            'mirostat_tau': 5,
            'mirostat_eta': 0.1,
            'guidance_scale': 1,
            'negative_prompt': '',

            'seed': -1,
            'add_bos_token': True,
            'truncation_length': 2048,
            'ban_eos_token': False,
            'skip_special_tokens': True,
            'stopping_strings': ["<END_OF_TURN>"]
        }

        request.update({
            'user_input': prompt, 
            'history': self.history.dict()
        })

        response = requests.post(URI, json=request)
        response.raise_for_status()

        result = response.json()

        ## dirty hack (needs to be fixed in API with stopping string)
        result = result['results'][0]['history']['visible'][-1][1]
        result = result.split('<END_OF_TURN>')
        result = result[0].strip()
        ## end dirty hack

        return result


class GenerateImageLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        URI = f'https://{HOST}/api/v1/chat'

        request = {
            'user_input': "prompt",
            'max_new_tokens': 60,
            'auto_max_new_tokens': False,
            'mode': 'instruct',  # Valid options: 'chat', 'chat-instruct', 'instruct'
            'character': 'None',
            'instruction_template': 'Orca-create-image2',  # Will get autodetected if unset
            #'your_name': 'You',
            #'name1': 'Idmon', # Optional
            #'name2': 'Hakeem', # Optional
            # 'context': 'character context', # Optional
            # 'greeting': 'greeting', # Optional
            # 'name1_instruct': 'You', # Optional
            # 'name2_instruct': 'Assistant', # Optional
            # 'context_instruct': 'context_instruct', # Optional
            # 'turn_template': 'turn_template', # Optional
            'regenerate': False,
            '_continue': False,
            'stop_at_newline': False,
            'chat_generation_attempts': 1,
            'chat_instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',

            # Generation params. If 'preset' is set to different than 'None', the values
            # in presets/preset-name.yaml are used instead of the individual numbers.
            'preset': 'None',
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.1,
            'top_k': 40,
            'typical_p': 1,
            'epsilon_cutoff': 0,  # In units of 1e-4
            'eta_cutoff': 0,  # In units of 1e-4
            'tfs': 1,
            'top_a': 0,
            'repetition_penalty': 1.05,
            'repetition_penalty_range': 0,
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'mirostat_mode': 0,
            'mirostat_tau': 5,
            'mirostat_eta': 0.1,
            'guidance_scale': 1,
            'negative_prompt': '',

            'seed': -1,
            'add_bos_token': True,
            'truncation_length': 2048,
            'ban_eos_token': False,
            'skip_special_tokens': True,
            'stopping_strings': []
        }

        request.update({
            'user_input': prompt
        })

        response = requests.post(URI, json=request)
        response.raise_for_status()

        result = response.json()

        return result['results'][0]['history']['visible'][-1][1]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}