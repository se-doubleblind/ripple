import json
import os
import re
import time
from typing import List

import google.generativeai as gemini

from dotenv import load_dotenv

from anthropic import Anthropic

from openai import AzureOpenAI


load_dotenv()


def extract_tag_list(
    tag: str,
    text: str,
    remove_leading_newline: bool = False,
) -> List[str]:
    """Extract a list of tags from a given XML string.

    Args:
        tag: The XML tag to extract.
        text: The XML string.
        remove_leading_newline: Whether to remove the leading newline from the
            extracted values.

    Returns:
        A list of values extracted from the provided tag.
    """
    # Define a regular expression pattern to match the tag
    pattern = rf"<{tag}(?:\s+[^>]*)?>(.*?)</{tag}>"

    # Use re.findall to extract all occurrences of the tag
    values = re.findall(pattern, text, re.DOTALL)

    if len(values) == 0:
        pattern = rf"<{tag}(?:\s+[^>]*)?>(.*)"
        values = re.findall(pattern, text, re.DOTALL)

    if remove_leading_newline:
        values = [v[1:] if v[0] == "\n" else v for v in values]
    return values


def extract_tag(
    tag: str,
    text: str,
    remove_leading_newline: bool = False,
) -> str:
    """Extract a tag from a given XML string.

    Args:
        tag: The XML tag to extract.
        text: The XML string.
        remove_leading_newline: Whether to remove the leading newline from the
            extracted values.

    Returns:
        An extracted string.
    """
    values = extract_tag_list(tag, text, remove_leading_newline)
    if len(values) > 0:
        return values[0]
    else:
        return ""


def process_response_with_change_plan(candidate_response):
    """
    """
    response = candidate_response.strip()
    change_plan = extract_tag('change_plan', response)
    return change_plan


def process_response_with_impact_set(candidate_response):
    """
    """
    response = candidate_response.strip()
    impacted_methods = [
        '.'.join(line.split(','))
        for line in extract_tag('impacted_methods', response).strip().split('\n')
    ]
    return impacted_methods


def process_response_with_impact_set_gemini(candidate_response):
    """
    """
    response = candidate_response.strip()
    impacted_methods = [
        '.'.join(line.split('.'))
        for line in extract_tag('impacted_methods', response).strip().split('\n')
    ]
    return impacted_methods


class AnthropicModel:
    """Helper class to invoke LLM from Anthropic"""
    def __init__(self, max_tokens_to_sample, temperature, top_p, num_responses):
        # `max_tokens_to_sample` can be atmost 4096 in claude-3-5-sonnet.
        self.model_name = 'claude-3-5-sonnet-20241022' #'claude-3-5-sonnet-20240620'
        self.inference_params = {
            "max_tokens_to_sample": max_tokens_to_sample,
            "temperature": temperature,
            "top_p": top_p,
        }
        self.num_responses = num_responses

    def _get_client(self):
        """Retrieve Anthropic client."""
        return Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), max_retries=25)

    def invoke(self, system: str, prompt: str, process_response_fn) -> str:
        """Invoke LLM with the given system and user prompts."""
        client = self._get_client()

        # Currently, Claude does not support multiple responses, and is not
        # directly suitable for self-exploration.
        all_candidates = []
        for _ in range(self.num_responses):
            start_time = time.time()
            response = client.messages.create(
                max_tokens=self.inference_params['max_tokens_to_sample'],
                system=system,
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=self.inference_params['temperature']
            )
            end_time = time.time()
            time.sleep(2)

            usage = {
                'completion_tokens': response.usage.output_tokens,
                'prompt_tokens': response.usage.input_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens,
                'time': end_time - start_time,
            }

            candidate_response = response.content[0].text
            candidate_set = process_response_fn(candidate_response)
            all_candidates.append((candidate_response, candidate_set, usage))

        return all_candidates


class GPTModel:
    def __init__(self, temperature, top_p, num_responses, seed):
        self.model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.inference_params = {
            "temperature": temperature,
            "top_p": top_p,
        }
        self.num_responses = num_responses
        self.seed = seed

    def _get_client(self):
        return AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    def invoke(self, system: str, prompt: str, process_response_fn) -> str:
        """Invoke LLM with the given system and user prompts."""
        client = self._get_client()

        start_time = time.time()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            seed=self.seed,
            temperature=self.inference_params['temperature'],
            top_p=self.inference_params['top_p'],
            n=self.num_responses,
            timeout = 240
        )
        end_time = time.time()
        usage = {
            'completion_tokens': response.usage.completion_tokens,
            'prompt_tokens': response.usage.prompt_tokens,
            'total_tokens': response.usage.total_tokens,
            'time': end_time - start_time,
        }

        all_candidates = []
        for candidate_id in range(self.num_responses):
            candidate_response = response.choices[candidate_id].message.content
            candidate_set = process_response_fn(candidate_response)
            all_candidates.append((candidate_response, candidate_set, usage))

        return all_candidates

    def invoke_with_tools(self, system: str, prompt: str, process_response_fn, tools) -> str:
        """Invoke LLM with the given system and user prompts."""
        client = self._get_client()

        start_time = time.time()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            seed=self.seed,
            temperature=self.inference_params['temperature'],
            top_p=self.inference_params['top_p'],
            n=self.num_responses,
            tools=tools,
            tool_choice='auto',
            timeout = 240
        )
        end_time = time.time()
        usage = {
            'completion_tokens': response.usage.completion_tokens,
            'prompt_tokens': response.usage.prompt_tokens,
            'total_tokens': response.usage.total_tokens,
            'time': end_time - start_time,
        }

        all_candidates = []
        for candidate_id in range(self.num_responses):
            candidate_response = response.choices[candidate_id].message.content
            candidate_set = process_response_fn(candidate_response)
            all_candidates.append((candidate_response, candidate_set, usage))

        return all_candidates


class GeminiModel:
    def __init__(self, temperature, top_p, num_responses, seed):
        self.model_name = "gemini-2.0-flash"
        self.inference_params = {
            "temperature": temperature,
            "top_p": top_p,
        }
        self.num_responses = num_responses
        self.seed = seed

    def _get_client(self, system):
        gemini.configure(api_key=os.getenv('GEMINI_API_KEY'))
        client = gemini.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system,
        )
        return client


    def invoke(self, system: str, prompt: str, process_response_fn) -> str:
        client = self._get_client(system)
        start_time = time.time()
        response = client.generate_content(
            prompt,
            generation_config=gemini.GenerationConfig(
                candidate_count=self.num_responses,
                temperature=self.inference_params['temperature'],
                top_p=self.inference_params['top_p'],
            ),
        )
        end_time = time.time()

        usage = {
            'completion_tokens': response.usage_metadata.candidates_token_count,
            'prompt_tokens': response.usage_metadata.prompt_token_count,
            'total_tokens': response.usage_metadata.total_token_count,
            'time': end_time - start_time,
        }

        all_candidates = []
        for candidate_id in range(self.num_responses):
            candidate_response = response.candidates[candidate_id].content.parts[0].text
            candidate = process_response_fn(candidate_response)
            print(candidate)
            print('='*10)
            all_candidates.append((candidate_response, candidate, usage))

        return all_candidates
