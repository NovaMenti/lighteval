# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from huggingface_hub import AsyncInferenceClient, InferenceTimeoutError
from openai import PermissionDeniedError
from pydantic import BaseModel, ValidationError
from requests.exceptions import HTTPError
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from lighteval.utils.imports import raise_if_package_not_available
from lighteval.utils.utils import as_list


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


DEFAULT_FORMAT = {"type": "text"}


@dataclass
class LitellmBackendOptions:
    """Options for the LiteLLM judge backend with default values.

    Attributes:
        caching (bool): Whether to enable caching for the API responses. Defaults to True.
        concurrent_requests (int): The maximum number of concurrent requests to the API. Defaults to 10.
        increase_max_tokens_for_reasoning (bool): Whether to increase the max tokens for certain reasoning
            models. Defaults to True.
        openrouter_provider_order (list[str] | None): For OpenRouter provider, restricts requests to ONLY the
            specified providers (e.g., ["parasail"]). This uses OpenRouter's "only" field to prevent fallback to
            other providers. Only used when model starts with "openrouter/". Defaults to None.
    """

    caching: bool = True
    concurrent_requests: int = 10

    # Increases max_tokens depending on the model used, see implementation below
    increase_max_tokens_for_reasoning: bool = True
    openrouter_provider_order: list[str] | None = None


class JudgeLM:
    """A class representing a judge for evaluating answers using either the chosen backend.

    Attributes:
        model (str): The name of the model.
        templates (Callable): A function taking into account the question, options, answer, and gold and returning the judge prompt.
        process_judge_response (Callable): A function for processing the judge's response.
        judge_backend (Literal["litellm", "openai", "transformers", "tgi", "vllm", "inference-providers"]): The backend for the judge.
        url (str | None): The URL for the OpenAI API.
        api_key (str | None): The API key for the OpenAI API (either OpenAI or HF key).
        max_tokens (int): The maximum number of tokens to generate. Defaults to 512.
        response_format (BaseModel | None): The format of the response from the API, used for the OpenAI and TGI backend.
        hf_provider (Literal["black-forest-labs", "cerebras", "cohere", "fal-ai", "fireworks-ai",
            "inference-providers", "hyperbolic", "nebius", "novita", "openai", "replicate", "sambanova", "together"] | None):
            The HuggingFace provider when using the inference-providers backend.
        backend_options (dict | None): Options for the backend. Currently only supported for litellm.

    Methods:
        evaluate_answer: Evaluates an answer using the OpenAI API or Transformers library.
        __lazy_load_client: Lazy loads the OpenAI client or Transformers pipeline.
        __call_api: Calls the API to get the judge's response.
        __call_transformers: Calls the Transformers pipeline to get the judge's response.
        __call_vllm: Calls the VLLM pipeline to get the judge's response.
    """

    def __init__(
        self,
        model: str,
        templates: Callable,
        process_judge_response: Callable,
        judge_backend: Literal["litellm", "openai", "transformers", "tgi", "vllm", "inference-providers"],
        url: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        response_format: BaseModel = None,
        hf_provider: Optional[
            Literal[
                "black-forest-labs",
                "cerebras",
                "cohere",
                "fal-ai",
                "fireworks-ai",
                "inference-providers",
                "hyperbolic",
                "nebius",
                "novita",
                "openai",
                "replicate",
                "sambanova",
                "together",
            ]
        ] = None,
        backend_options: dict | None = None,
    ):
        self.model = model
        self.template = templates

        self.API_MAX_RETRY = 3
        self.API_RETRY_SLEEP = 1

        self.client = None
        self.pipe = None
        self.process_judge_response = process_judge_response

        self.url = url
        self.api_key = api_key
        self.backend = judge_backend
        self.hf_provider = hf_provider
        self.max_tokens = max_tokens

        self.response_format = response_format if response_format else DEFAULT_FORMAT

        self.backend_options = backend_options or {}

        # Override backend options dictionary with the corresponding dataclass to ensure all specified options are valid
        if judge_backend == "litellm":
            self.backend_options = LitellmBackendOptions(**self.backend_options)

        # Validate that hf_provider is specified when using inference-providers backend
        if self.backend == "inference-providers" and self.hf_provider is None:
            raise ValueError("When using 'inference-providers' as backend, you must specify an 'hf_provider'")

    def __lazy_load_client(self):  # noqa: C901
        match self.backend:
            # Both "openai" and "tgi" backends use the OpenAI-compatible API
            # They are handled separately to allow for backend-specific validation and setup
            case "openai" | "tgi":
                raise_if_package_not_available("openai")
                if self.client is None:
                    from openai import OpenAI

                    self.client = OpenAI(
                        api_key=self.api_key if self.url is None else None,
                        base_url=self.url if self.url else None,
                    )
                return self.__call_api_parallel

            case "litellm":
                raise_if_package_not_available("litellm")
                return self.__call_litellm

            case "vllm":
                raise_if_package_not_available("vllm")
                if self.pipe is None:
                    from vllm import LLM, SamplingParams
                    from vllm.transformers_utils.tokenizer import get_tokenizer

                    self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=self.max_tokens)
                    self.tokenizer = get_tokenizer(self.model, tokenizer_mode="auto")
                    self.pipe = LLM(model=self.model, gpu_memory_utilization=0.8, dtype="float16")
                return self.__call_vllm

            case "transformers":
                if self.pipe is None:
                    import torch
                    from transformers import (
                        AutoModelForCausalLM,
                        AutoTokenizer,
                        pipeline,
                    )

                    transformers_model = AutoModelForCausalLM.from_pretrained(
                        self.model,
                        torch_dtype=torch.float16,
                        trust_remote_code=False,
                        device_map="cuda",
                    )
                    tokenizer = AutoTokenizer.from_pretrained(self.model)
                    self.pipe = pipeline(
                        "text-generation",
                        model=transformers_model,
                        tokenizer=tokenizer,
                        max_new_tokens=self.max_tokens,
                    )
                return self.__call_transformers

            case "inference-providers":
                from huggingface_hub import AsyncInferenceClient

                self.client = AsyncInferenceClient(token=self.api_key, base_url=self.url, provider=self.hf_provider)
                return self.__call_hf_inference_async

            case _:
                raise ValueError(f"Unsupported backend: {self.backend}")

    def dict_of_lists_to_list_of_dicts(self, dict_of_lists):
        """Transform a dictionary of lists into a list of dictionaries.

        Each dictionary in the output list will contain one element from each list in the input dictionary,
        with the same keys as the input dictionary.

        Args:
            dict_of_lists: A dictionary where each value is a list.
                           All lists are expected to have the same length.

        Returns:
            A list of dictionaries.

        Example:
            >>> dict_of_lists_to_list_of_dicts({'k': [1, 2, 3], 'k2': ['a', 'b', 'c']})
            [{'k': 1, 'k2': 'a'}, {'k': 2, 'k2': 'b'}, {'k': 3, 'k2': 'c'}]
        """
        # Check if input is empty
        if not dict_of_lists:
            return None

        # Get all list lengths to ensure they match
        list_lengths = [len(values) for values in dict_of_lists.values()]

        # Ensure all lists have the same length
        if len(set(list_lengths)) > 1:
            raise ValueError("All lists in the input dictionary must have the same length")

        # Get the length of the lists
        n = list_lengths[0] if list_lengths else 0

        # Create list of dictionaries
        result = []
        for i in range(n):
            new_dict = {key: values[i] for key, values in dict_of_lists.items()}
            result.append(new_dict)

        return result

    def evaluate_answer_batch(
        self,
        questions: list[str],
        answers: list[str],
        options: list[list[str]] | list[None],
        golds: list[str] | list[None],
        **kwargs,
    ):
        judge_function = self.__lazy_load_client()

        kwargss = self.dict_of_lists_to_list_of_dicts(kwargs)
        if kwargss is None:
            kwargss = [{} for _ in range(len(questions))]

        # enumerate over questions answers options and golds to make the
        prompts = [
            self.template(question=q, answer=a, options=o, gold=g, **k)
            for q, a, o, g, k in zip(questions, answers, options, golds, kwargss)
        ]
        responses = judge_function(prompts)
        scores = [self.process_judge_response(response) for response in responses]

        # clean up the vllm pipeline and free up memory
        if self.pipe is not None and self.backend == "vllm":
            del self.pipe
            self.pipe = None

        return scores, prompts, responses

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        options: list[str] | None = None,
        gold: str | None = None,
    ):
        """Evaluates an answer using either Transformers or OpenAI API.

        Args:
            question (str): The prompt asked to the evaluated model.
            answer (str): Answer given by the evaluated model.
            options (list[str] | None): Optional list of answer options.
            gold (str | None): Optional reference answer.

        Returns:
            A tuple containing the score, prompts, and judgment.
        """
        # lazy loading of the pipeline
        judge_function = self.__lazy_load_client()
        prompt = self.template(question=question, options=options, answer=answer, gold=gold)
        response = judge_function(prompt)
        score = self.process_judge_response(response)

        return score, prompt, response

    def __call_transformers(self, prompt):
        response = self.pipe(prompt)[0]["generated_text"]
        response = response[-1]["content"]
        return response

    def __call_vllm(self, prompt):
        tokenized = [self.tokenizer.apply_chat_template(p) for p in prompt]
        output = self.pipe.generate(
            prompt_token_ids=tokenized,
            sampling_params=self.sampling_params,
            use_tqdm=True,
        )
        outputs = [output.outputs[0].text for output in output]
        return outputs

    def __call_litellm(self, prompts):  # noqa: C901
        import litellm

        if self.backend_options.caching:
            from litellm.caching.caching import Cache, LiteLLMCacheType

            litellm.cache = Cache(type=LiteLLMCacheType.DISK)

        # Automatically drop parameters that are not supported by the currently used inference API
        litellm.drop_params = True

        def _classify_error(error: Exception, attempt: int) -> dict:
            """Classify error and determine retry strategy.

            Returns a dict with:
                - should_retry: bool indicating if we should retry
                - message: str error message prefix
                - log_func: callable logger function (logger.error, logger.warning, etc.)
                - hint: str optional hint for fixing the issue
            """
            error_type = type(error)

            # Non-retryable errors (fail fast)
            NON_RETRYABLE = {
                litellm.ContentPolicyViolationError: {
                    "message": "CONTENT POLICY VIOLATION: Request blocked by provider content filters.",
                    "hint": "Failing fast and returning empty response.",
                    "log_func": logger.error,
                },
                litellm.ContextWindowExceededError: {
                    "message": f"CONTEXT WINDOW EXCEEDED: Input exceeds model context window (max_model_length={self.max_length}).",
                    "hint": "Reduce input size or increase max_model_length.",
                    "log_func": logger.error,
                },
                litellm.UnsupportedParamsError: {
                    "message": "UNSUPPORTED PARAMETERS: Invalid parameters passed to API.",
                    "hint": "Check model configuration.",
                    "log_func": logger.error,
                },
                litellm.AuthenticationError: {
                    "message": "AUTHENTICATION ERROR: Invalid API key or authentication failed.",
                    "hint": "Check API credentials.",
                    "log_func": logger.error,
                },
                PermissionDeniedError: {
                    "message": "PERMISSION DENIED: Insufficient permissions for this request.",
                    "hint": "Check API key permissions.",
                    "log_func": logger.error,
                },
                litellm.NotFoundError: {
                    "message": f"MODEL NOT FOUND: Invalid model name or model not available (Model: {self.model}).",
                    "hint": "Check model name.",
                    "log_func": logger.error,
                },
                litellm.UnprocessableEntityError: {
                    "message": "UNPROCESSABLE ENTITY: Request format is invalid.",
                    "hint": "Check request parameters.",
                    "log_func": logger.error,
                },
                litellm.BudgetExceededError: {
                    "message": "BUDGET EXCEEDED: API budget limit reached.",
                    "hint": "Check account budget settings.",
                    "log_func": logger.error,
                },
                litellm.BadRequestError: {
                    "message": "BAD REQUEST: Invalid request parameters.",
                    "hint": "Check request format and parameters.",
                    "log_func": logger.error,
                },
            }

            # Special case: JSONSchemaValidationError - retry once
            if error_type == litellm.JSONSchemaValidationError:
                if attempt == 0:
                    return {
                        "should_retry": True,
                        "message": "JSON SCHEMA VALIDATION ERROR: Response does not match expected schema. Retrying once...",
                        "log_func": logger.warning,
                    }
                return {
                    "should_retry": False,
                    "message": "JSON SCHEMA VALIDATION ERROR: Response does not match expected schema.",
                    "hint": "This may indicate a model/provider issue.",
                    "log_func": logger.warning,
                }

            # Check non-retryable errors
            if error_type in NON_RETRYABLE:
                return {"should_retry": False, **NON_RETRYABLE[error_type]}

            # Retryable errors (transient network/server issues)
            RETRYABLE = {
                litellm.Timeout: "TIMEOUT ERROR: Request timed out.",
                litellm.RateLimitError: "RATE LIMIT ERROR: API rate limit exceeded.",
                litellm.APIConnectionError: "API CONNECTION ERROR: Failed to connect to API.",
                litellm.ServiceUnavailableError: "SERVICE UNAVAILABLE: API service is temporarily unavailable.",
                litellm.InternalServerError: "INTERNAL SERVER ERROR: API server encountered an error.",
                litellm.APIError: "API ERROR: Generic API error occurred.",
            }

            if error_type in RETRYABLE:
                return {
                    "should_retry": True,
                    "message": RETRYABLE[error_type],
                    "log_func": logger.warning,
                }

            # Unknown exception - retry but log as error
            return {
                "should_retry": True,
                "message": f"UNKNOWN ERROR: Unexpected exception type {error_type.__name__}.",
                "log_func": logger.error,
            }

        def _check_finish_reason(
            choices,
        ) -> None:
            # Check for finish_reason issues and log appropriate warnings/errors
            for i, choice in enumerate(choices):
                finish_reason = getattr(choice, "finish_reason", None)
                native_finish_reason = getattr(choice, "native_finish_reason", None)

                if finish_reason == "length":
                    # Truncation: response incomplete due to token limit
                    native_info = (
                        f" (native: {native_finish_reason})"
                        if native_finish_reason and native_finish_reason != finish_reason
                        else ""
                    )
                    logger.warning(
                        f"TRUNCATION DETECTED: Response {i + 1} was truncated due to token limit{native_info}. "
                        f"Consider increasing max_new_tokens or max_model_length to ensure complete responses."
                    )
                elif finish_reason == "content_filter":
                    # Content was filtered: response may be incomplete or missing
                    native_info = (
                        f" (native: {native_finish_reason})"
                        if native_finish_reason and native_finish_reason != finish_reason
                        else ""
                    )
                    logger.warning(
                        f"CONTENT FILTERED: Response {i + 1} was filtered by content moderation{native_info}. "
                        f"Response may be incomplete or missing. Review the prompt or model settings."
                    )
                elif finish_reason == "error":
                    # Error occurred: response likely incomplete or missing
                    native_info = (
                        f" (native: {native_finish_reason})"
                        if native_finish_reason and native_finish_reason != finish_reason
                        else ""
                    )
                    logger.error(
                        f"GENERATION ERROR: Response {i + 1} encountered an error during generation{native_info}. "
                        f"Response may be incomplete or missing. Check model/provider status."
                    )
                elif finish_reason == "tool_calls":
                    # Tool calls: normal for function calling, but unexpected here
                    native_info = (
                        f" (native: {native_finish_reason})"
                        if native_finish_reason and native_finish_reason != finish_reason
                        else ""
                    )
                    logger.info(
                        f"TOOL CALLS: Response {i + 1} stopped for tool/function calls{native_info}. "
                        f"This is unexpected if not using function calling."
                    )
                elif finish_reason and finish_reason != "stop":
                    # Unknown finish reason
                    native_info = f" (native: {native_finish_reason})" if native_finish_reason else ""
                    logger.info(f"Response {i + 1} finished with reason: {finish_reason}{native_info}")

        def _clean_response(text: str) -> str | BaseModel:
            if (
                self.response_format is not None
                and isinstance(self.response_format, type)
                and issubclass(self.response_format, BaseModel)
            ):
                import json
                import re

                # Strip markdown code blocks if present
                cleaned_text = text.strip()
                if "```" in cleaned_text:
                    cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text, flags=re.MULTILINE)
                    cleaned_text = re.sub(r"\s*```\s*$", "", cleaned_text, flags=re.MULTILINE)

                try:
                    parsed_json = json.loads(cleaned_text)
                    validated_model = self.response_format.model_validate(parsed_json)
                    return validated_model  # Return Pydantic model instead of string
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.warning(f"Failed to parse structured response: {e}, returning raw text")
                    return text
                except Exception as e:
                    logger.warning(f"Unexpected error during response validation: {e}, returning raw text")
                    return text
            else:
                return text

        def __call_api(prompt):
            error_message = "ERROR: Failed to get response from the API."
            for attempt in range(self.API_MAX_RETRY):
                try:
                    max_new_tokens = self.max_tokens

                    is_reasoning_model = "o1" in self.model or "o3" in self.model or "R1" in self.model
                    if is_reasoning_model and self.backend_options.increase_max_tokens_for_reasoning:
                        max_new_tokens = min(max_new_tokens * 10, 32000)

                    kwargs = {
                        "model": self.model,
                        "messages": prompt,
                        "n": 1,
                        "caching": True,
                    }
                    if self.response_format is not None:
                        kwargs["response_format"] = self.response_format
                    if max_new_tokens is not None:
                        kwargs["max_tokens"] = (max_new_tokens,)

                    response = litellm.completion(**kwargs)
                    text = response.choices[0].message.content
                    if not text or text == error_message:
                        logger.info(f"Retrying without caching for prompt: {prompt[:100]}...")
                        retry_kwargs = {**kwargs, "caching": False}
                        response = litellm.completion(**retry_kwargs)
                        text = _clean_response(response.choices[0].message.content)
                        if not text or text == error_message:
                            # Just return None if the second attempt fails too
                            logger.error(f"Failed to get response from the API for prompt: {prompt[:100]}...")
                            return None

                    _check_finish_reason(response.choices)
                    cleaned_text = _clean_response(text)
                    return cleaned_text
                except Exception as e:
                    error_action = _classify_error(e, attempt)
                    if error_action["should_retry"]:
                        wait_time = min(
                            64,
                            self.API_RETRY_SLEEP * (2.0**attempt),
                        )
                        error_action["log_func"](
                            f"{error_action['message']} "
                            f"Error: {str(e)}, waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        error_action["log_func"](
                            f"{error_action['message']} Error: {str(e)}. {error_action.get('hint', '')}"
                        )
                        return None
            logger.error(f"All {self.API_MAX_RETRY} retry attempts exhausted. Returning empty response.")
            return error_message

        results = []
        with ThreadPoolExecutor(self.backend_options.concurrent_requests) as executor:
            for entry in tqdm(executor.map(__call_api, prompts), total=len(prompts)):
                results.append(entry)

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")

        return results

    def __call_hf_inference_async(self, prompts):
        async def run_all() -> list[str]:
            """Wrap inference call into function"""
            tasks = (self.__call_hf_inference(prompt) for prompt in prompts)
            return await tqdm_asyncio.gather(*tasks, desc="HF inference", total=len(prompts))

        try:
            loop = asyncio.get_running_loop()
            logger.debug("Exting event loop is found, using loop.create_task")
            result = loop.run_until_complete(run_all())
        except RuntimeError:
            logger.debug("No running event loop found, using asyncio.run")
            result = asyncio.run(run_all())

        if None in result:
            logger.warning("None found in inference results")

        return result

    async def __call_hf_inference(self, prompt):
        self.client: AsyncInferenceClient
        for _ in range(self.API_MAX_RETRY):
            try:
                response = await self.client.chat_completion(
                    model=self.model,
                    messages=prompt,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except (InferenceTimeoutError, HTTPError) as e:
                logger.warning(f"HTTP error during HF inference: {e}")
                await asyncio.sleep(self.API_RETRY_SLEEP)
            except Exception as e:
                logger.warning(f"Unexpected error during HF inference: {e}")
                await asyncio.sleep(self.API_RETRY_SLEEP)

        raise Exception("Failed to get response from the HF API")

    def __call_api_parallel(self, prompts):
        results = []
        with ThreadPoolExecutor(10) as executor:
            for entry in tqdm(executor.map(self.__call_api, prompts), total=len(prompts)):
                results.append(entry)

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")

        return results

    def __call_api(self, prompt):
        for _ in range(self.API_MAX_RETRY):
            try:
                # Base model
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=as_list(prompt),
                    response_format=self.response_format,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    n=1,
                )
                answer = response.choices[0].message.parsed
                return answer
            except TypeError:
                try:
                    # Finetune
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=as_list(prompt),
                        response_format=self.response_format,
                        max_tokens=self.max_tokens,
                        n=1,
                    )
                    text = response.choices[0].message.content
                    return text
                except Exception as e:
                    logger.warning(f"{type(e), e}")
                    time.sleep(self.API_RETRY_SLEEP)
            except Exception as e:
                logger.warning(f"{type(e), e}")
                time.sleep(self.API_RETRY_SLEEP)

        raise Exception("Failed to get response from the API")

    def __str__(self) -> str:
        return f"Model: {self.model}, Judge Backend: {self.backend}, URL: {self.url}"
