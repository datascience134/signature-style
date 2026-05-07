import json
import re

from openai import AzureOpenAI, OpenAI
from core import constants


class LLMInterface:
    """
    Azure AI Foundry OpenAI-compatible client for text chat completions.
    """

    def __init__(self):
        self.client = OpenAI(
            base_url=constants.AZUREFOUNDRY_ENDPOINT,
            api_key=constants.AZUREFOUNDRY_APIKEY,
        )  # for gpt-5.4
        # self.client = AzureOpenAI(
        #     azure_endpoint=constants.AZUREOPENAI_ENDPOINT,
        #     api_key=constants.AZUREOPENAI_APIKEY,
        #     api_version=constants.AZUREOPENAI_API_VERION,
        # )  # for gpt-4.1

    def llm_text(
        self,
        system_prompt: str,
        user_content: str,
        response_format: dict = {"type": "json_object"},
        temperature: float = 0,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stop=None,
    ):
        response = self.client.chat.completions.create(
            model=constants.GPT5_4_DEPLOYMENT_NAME,  # for gpt-5.4
            # model=constants.AZUREOPENAI_MODEL,  # for gpt-4.1
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )

        return response.choices[0].message.content

    def post_process_llm_response(self, processing_prompt: str, response_content: str):
        try:
            cleaned_response = json.loads(response_content)
            return cleaned_response
        except json.JSONDecodeError:
            cleaned = re.sub(
                r"^```(?:json)?\n|```$",
                "",
                response_content.strip(),
                flags=re.MULTILINE,
            )
            cleaned_response = json.loads(cleaned)
            return cleaned_response
