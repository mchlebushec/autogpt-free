from __future__ import annotations


import tiktoken

from typing import List, Optional
import requests

import openai
from openai import Model
import json
from autogpt.config import Config
from autogpt.llm.base import MessageDict
from autogpt.llm.modelsinfo import COSTS
from autogpt.logs import logger
from autogpt.singleton import Singleton


class ApiManager(metaclass=Singleton):
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0
        self.models: Optional[list[Model]] = None

    def reset(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0.0
        self.models = None

    def num_tokens_from_string(self, string: str, encoding_model: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(encoding_model)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def create_chat_completion(
        self,
        messages: list[MessageDict],
        model: str | None = None,
        temperature: float = None,
        max_tokens: int | None = None,
        deployment_id=None,
    ) -> str:
        """
        Create a chat completion and update the cost.
        Args:
        messages (list): The list of messages to send to the API.
        model (str): The model to use for the API call.
        temperature (float): The temperature to use for the API call.
        max_tokens (int): The maximum number of tokens for the API call.
        Returns:
        str: The AI's response.
        """
        cfg = Config()
        if temperature is None:
            temperature = cfg.temperature
        if True:
            messages_str = ''
            for i in messages:
                messages_str += i['role'] + ": " + i['content'] + "\n"
            data = json.dumps({
                'prompt': messages_str,
                'max_tokens': max_tokens,
            })
            headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/110.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Content-Type": "application/json",
    "Origin": "https://chatbot.theb.ai",
    "Referer": "https://chatbot.theb.ai/"
}
            url = "https://chatbot.theb.ai/api/chat-process"
            response = '{' + requests.post(url, data=data, headers=headers).text.split("{", 1)[1]
            json_strings = response.strip().split('\n')
            last_json_string = json_strings[-1]
            response = json.loads(last_json_string)['text']
        if response:
            logger.debug(f"Response: {response}")
            prompt_tokens = self.num_tokens_from_string(messages_str, 'gpt-3.5-turbo')
            completion_tokens = self.num_tokens_from_string(response, 'gpt-3.5-turbo')
            self.update_cost(prompt_tokens, completion_tokens, model)
        return response

    def update_cost(self, prompt_tokens, completion_tokens, model: str):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        # the .model property in API responses can contain version suffixes like -v2
        model = model[:-3] if model.endswith("-v2") else model

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += (
            prompt_tokens * COSTS[model]["prompt"]
            + completion_tokens * COSTS[model]["completion"]
        ) / 1000
        logger.debug(f"Total running cost: ${self.total_cost:.3f}")

    def set_total_budget(self, total_budget):
        """
        Sets the total user-defined budget for API calls.

        Args:
        total_budget (float): The total budget for API calls.
        """
        self.total_budget = total_budget

    def get_total_prompt_tokens(self):
        """
        Get the total number of prompt tokens.

        Returns:
        int: The total number of prompt tokens.
        """
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        """
        Get the total number of completion tokens.

        Returns:
        int: The total number of completion tokens.
        """
        return self.total_completion_tokens

    def get_total_cost(self):
        """
        Get the total cost of API calls.

        Returns:
        float: The total cost of API calls.
        """
        return self.total_cost

    def get_total_budget(self):
        """
        Get the total user-defined budget for API calls.

        Returns:
        float: The total budget for API calls.
        """
        return self.total_budget

    def get_models(self) -> List[Model]:
        """
        Get list of available GPT models.

        Returns:
        list: List of available GPT models.

        """
        if self.models is None:
            all_models = openai.Model.list()["data"]
            self.models = [model for model in all_models if "gpt" in model["id"]]

        return self.models
