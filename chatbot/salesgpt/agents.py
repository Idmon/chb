import re
from copy import deepcopy
from typing import Any, Dict, List, Union
import logging

from langchain import LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from pydantic import BaseModel, Field

from salesgpt.chains import SalesConversationChain, StageAnalyzerChain, ImagePromptAnalyzerChain, ChatAnalyzerChain, ConversationAnalyzerChain
from salesgpt.logger import time_logger
from salesgpt.parsers import SalesConvoOutputParser
from salesgpt.prompts import SALES_AGENT_TOOLS_PROMPT
from salesgpt.stages import CONVERSATION_STAGES
from salesgpt.templates import CustomPromptTemplateForTools
from salesgpt.tools import get_tools, setup_knowledge_base
from salesgpt.customLLM import GenerateImageLLM

logger = logging.getLogger("Agent")

class SalesGPT(Chain, BaseModel):
    """Controller model for the Sales Agent."""

    face: str = "default"

    conversation_history: List[str] = []
    conversation_stage_id: str = "1"
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    image_analyzer_chain: ImagePromptAnalyzerChain = Field(...)
    conversation_analyzer_chain: ConversationAnalyzerChain = Field(...)
    chat_analyzer_chain: ChatAnalyzerChain = Field(...)
    sales_agent_executor: Union[AgentExecutor, None] = Field(...)
    knowledge_base: Union[RetrievalQA, None] = Field(...)
    sales_conversation_utterance_chain: SalesConversationChain = Field(...)
    conversation_stage_dict: Dict = CONVERSATION_STAGES
    conversation_stages="\n".join(
        [
            str(key) + ": " + str(value)
            for key, value in conversation_stage_dict.items()
        ]
    )
    current_conversation_stage: str = conversation_stage_dict.get("1")
    image_options: Dict[str, Dict[str, Union[str, List[str]]]] = Field(...)

    use_tools: bool = False
    tools: List[str] = []
    salesperson_name: str = "Ted Lasso"
    salesperson_role: str = "Business Development Representative"
    company_name: str = "Sleep Haven"
    company_business: str = "Business description"
    company_values: str = "Company Values"
    conversation_purpose: str = "Sell mattress to the prospect and help him out during the conversation"
    conversation_type: str = "call"

    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, "1")

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    @time_logger
    def seed_agent(self):
        # Step 1: seed the conversation
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []
    
    @staticmethod
    def parse_output(output_str):
        # Extract the content before <SCENE>
        scene_match = re.search(r'^(.*?)<SCENE>', output_str)
        # Extract the content after <SCENE> but before <CLOTHING>
        clothing_match = re.search(r'<SCENE>(.*?)<CLOTHING>', output_str)
        
        scene_str = scene_match.group(1).strip() if scene_match else ""
        clothing_str = clothing_match.group(1).strip() if clothing_match else ""
        
        return scene_str, clothing_str

    @staticmethod
    def parse_output2(output_str):
        parts = [part.strip() for part in output_str.split(',')]
        key = parts[0].strip().replace(" ", "")
        values = parts[1:]
        return {key: values}


    def construct_prompt(self, image_instructions):
        for key, values in image_instructions.items():
            if key in self.image_options:
                prompt = self.image_options[key]['prompt']
                negPrompt = self.image_options[key]['negative_prompt']
                
                input_fields = self.image_options[key]['input']
                
                # If image_instructions has more values than input_fields, prune them.
                values = values[:len(input_fields)]
                
                # If image_instructions has fewer values than input_fields, fill in with empty strings.
                while len(values) < len(input_fields):
                    values.append("")
                
                for idx, value in enumerate(values):
                    placeholder = "{" + input_fields[idx] + "}"
                    prompt = prompt.replace(placeholder, value)
                    print(prompt)
                
                return prompt, negPrompt, self.face

        # If the loop completes without returning, then return None.
        return None


    @time_logger
    def create_image_prompt(self):
        last_messages = "\n".join(self.conversation_history[-10:]).rstrip("\n")
        conversationContext = self.conversation_analyzer_chain.run(
            conversation_history=last_messages,
        )

        image_options_str = "\n".join(
            [
                f"{key} [{', '.join(value['input'])}]" if value['input'] else key
                for key, value in self.image_options.items()
            ]
        )

        image_prompt = self.image_analyzer_chain.run(
            image_options=image_options_str,
            conversation_context=conversationContext
        ).lower()
        logger.info(f"Output from LLM: {image_prompt}")
        image_instructions = self.parse_output2(image_prompt)
        logger.info(f"Parsed prompt instructions: {image_prompt}")

        return image_instructions


    @time_logger
    def determine_conversation_stage(self):
        self.conversation_stage_id = self.stage_analyzer_chain.run(
            conversation_history="\n".join(self.conversation_history).rstrip("\n"),
            conversation_stage_id=self.conversation_stage_id,
            conversation_stages="\n".join(
                [
                    str(key) + ": " + str(value)
                    for key, value in conversation_stage_dict.items()
                ]
            ),
        )

        logger.info(f"Conversation Stage ID: {self.conversation_stage_id}")
        self.current_conversation_stage = self.retrieve_conversation_stage(
            self.conversation_stage_id
        )

        logger.info(f"Conversation Stage: {self.current_conversation_stage}")

    def human_step(self, human_input):
        # process human input
        human_input = "User: " + human_input + " <END_OF_TURN>"
        self.conversation_history.append(human_input)
    
    def system_step(self, system_input):
        # process system input
        human_input = "SYSTEM: " + system_input
        self.conversation_history.append(human_input)

    @time_logger
    def step(
        self, return_streaming_generator: bool = False, model_name="gpt-3.5-turbo-0613", image=False
    ):
        """
        Args:
            return_streaming_generator (bool): whether or not return
            streaming generator object to manipulate streaming chunks in downstream applications.
        """
        if not return_streaming_generator:
            return self._call(inputs={})
        else:
            return self._streaming_generator(model_name=model_name)

    # TO-DO change this override "run" override the "run method" in the SalesConversation chain!
    @time_logger
    def _streaming_generator(self, model_name="gpt-3.5-turbo-0613"):
        """
        Sometimes, the sales agent wants to take an action before the full LLM output is available.
        For instance, if we want to do text to speech on the partial LLM output.

        This function returns a streaming generator which can manipulate partial output from an LLM
        in-flight of the generation.

        Example:

        >> streaming_generator = self._streaming_generator()
        # Now I can loop through the output in chunks:
        >> for chunk in streaming_generator:
        Out: Chunk 1, Chunk 2, ... etc.
        See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
        """
        prompt = self.sales_conversation_utterance_chain.prep_prompts(
            [
                dict(
                    conversation_stage=self.current_conversation_stage,
                    conversation_history="\n".join(self.conversation_history),
                    salesperson_name=self.salesperson_name,
                    salesperson_role=self.salesperson_role,
                    company_name=self.company_name,
                    company_business=self.company_business,
                    company_values=self.company_values,
                    conversation_purpose=self.conversation_purpose,
                    conversation_type=self.conversation_type,
                    conversation_stages=self.conversation_stages,
                    conversation_stage_dict=self.conversation_stage_dict,
                    tools=self.tools
                )
            ]
        )

        inception_messages = prompt[0][0].to_messages()

        message_dict = {"role": "system", "content": inception_messages[0].content}

        if self.sales_conversation_utterance_chain.verbose:
            print("\033[92m" + inception_messages[0].content + "\033[0m")
        messages = [message_dict]

        return self.sales_conversation_utterance_chain.llm.completion_with_retry(
            messages=messages,
            stop="<END_OF_TURN>",
            stream=True,
            model=model_name,
        )

    def _call(self, inputs: Dict[str, Any]) -> None:
        """Run one step of the sales agent."""

        last_messages = "\n".join(self.conversation_history[-10:])

        # Generate agent's utterance
        # if use tools
        if self.use_tools:
            print("AGENT")
            last_message = self.conversation_history[-1]
            ai_message = self.chat_analyzer_chain.run(
                answer=last_messages
            )
            logger.info(f"Answer from LLM: {ai_message}")


            # ai_message = self.sales_agent_executor.run(
            #     input="",
            #     conversation_stage=self.current_conversation_stage,
            #     conversation_history=last_messages,
            #     salesperson_name=self.salesperson_name,
            #     salesperson_role=self.salesperson_role,
            #     company_name=self.company_name,
            #     company_business=self.company_business,
            #     company_values=self.company_values,
            #     conversation_purpose=self.conversation_purpose,
            #     conversation_type=self.conversation_type,
            #     conversation_stages=self.conversation_stages,
            #     conversation_stage_dict=self.conversation_stage_dict,
            #     tools=self.tools
            # )

        else:
            # else
            ai_message = self.sales_conversation_utterance_chain.run(
                conversation_stage=self.current_conversation_stage,
                conversation_history=last_messages,
                salesperson_name=self.salesperson_name,
                salesperson_role=self.salesperson_role,
                company_name=self.company_name,
                company_business=self.company_business,
                company_values=self.company_values,
                conversation_purpose=self.conversation_purpose,
                conversation_type=self.conversation_type,
                conversation_stages=self.conversation_stages,
                conversation_stage_dict=self.conversation_stage_dict,
                tools=self.tools
            )

        # Add agent's response to conversation history
        message = ai_message.replace("<END_OF_TURN>", "")
        if '<END_OF_TURN>' not in ai_message:
            ai_message += ' <END_OF_TURN>'
        agent_name = self.salesperson_name
        ai_message = agent_name + ": " + ai_message
        self.conversation_history.append(ai_message)
        logger.info(agent_name + ": " + message)
        return message


    @classmethod
    @time_logger
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "SalesGPT":
        """Initialize the SalesGPT Controller."""
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)
        llmImage = GenerateImageLLM()
        image_analyzer_chain = ImagePromptAnalyzerChain.from_llm(llmImage, verbose=verbose)
        conversation_analyzer_chain = ConversationAnalyzerChain.from_llm(llmImage, verbose=verbose)
        chat_analyzer_chain = ChatAnalyzerChain.from_llm(llm, verbose=verbose)
        if (
            "use_custom_prompt" in kwargs.keys()
            and kwargs["use_custom_prompt"] == "True"
        ):
            use_custom_prompt = deepcopy(kwargs["use_custom_prompt"])
            custom_prompt = deepcopy(kwargs["custom_prompt"])

            # clean up
            del kwargs["use_custom_prompt"]
            del kwargs["custom_prompt"]

            sales_conversation_utterance_chain = SalesConversationChain.from_llm(
                llm,
                verbose=verbose,
                use_custom_prompt=use_custom_prompt,
                custom_prompt=custom_prompt
            )

        else:
            sales_conversation_utterance_chain = SalesConversationChain.from_llm(
                llm, verbose=verbose
            )

        if "use_tools" in kwargs.keys() and kwargs["use_tools"] is True:
            # set up agent with tools
            load_tools = kwargs.get("tools", '')
            product_catalog = kwargs["product_catalog"]
            knowledge_base = setup_knowledge_base(product_catalog)
            tools = get_tools(knowledge_base, load_tools)

            prompt = CustomPromptTemplateForTools(
                template = custom_prompt if custom_prompt is not None else SALES_AGENT_TOOLS_PROMPT,
                tools_getter=lambda x: tools,
                input_variables=[
                    "input",
                    "intermediate_steps",
                    "salesperson_name",
                    "salesperson_role",
                    "company_name",
                    "company_business",
                    "company_values",
                    "conversation_purpose",
                    "conversation_type",
                    "conversation_history",
                    "conversation_stages",
                    "tools"
                ],
            )
            
            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)

            tool_names = [tool.name for tool in tools]

            output_parser = SalesConvoOutputParser(ai_prefix=kwargs["salesperson_name"])

            sales_agent_with_tools = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names,
            )

            sales_agent_executor = AgentExecutor.from_agent_and_tools(
                agent=sales_agent_with_tools, tools=tools, verbose=verbose
            )
        else:
            sales_agent_executor = None
            knowledge_base = None

        conversation_stage_dict = kwargs.pop("conversation_stage_dict")
        conversation_stages="\n".join(
            [
                str(key) + ": " + str(value)
                for key, value in conversation_stage_dict.items()
            ]
        )

        return cls(
            conversation_analyzer_chain=conversation_analyzer_chain,
            chat_analyzer_chain=chat_analyzer_chain,
            conversation_stages=conversation_stages,
            conversation_stage_dict=conversation_stage_dict,
            stage_analyzer_chain=stage_analyzer_chain,
            image_analyzer_chain=image_analyzer_chain,
            sales_conversation_utterance_chain=sales_conversation_utterance_chain,
            sales_agent_executor=sales_agent_executor,
            knowledge_base=knowledge_base,
            verbose=verbose,
            **kwargs,
        )

