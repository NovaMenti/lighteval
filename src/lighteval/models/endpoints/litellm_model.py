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

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from typing import Any, List

import requests
from openai import PermissionDeniedError
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached
from lighteval.utils.imports import is_package_available, requires


logger = logging.getLogger(__name__)

if is_package_available("litellm"):
    import litellm
    from litellm import encode, supports_reasoning
    from litellm.caching.caching import Cache, LiteLLMCacheType
    from litellm.utils import ModelResponse as LitellmModelResponse
    from litellm.utils import get_max_tokens

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").handlers.clear()

    litellm.cache = Cache(type=LiteLLMCacheType.DISK)
else:
    from unittest.mock import Mock

    litellm = Mock()
    encode = Mock()
    LitellmModelResponse = Mock()

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None


class LiteLLMModelConfig(ModelConfig):
    """Configuration class for LiteLLM unified API client.

    This configuration is used to connect to various LLM providers through the LiteLLM
    unified API. LiteLLM provides a consistent interface to multiple providers including
    OpenAI, Anthropic, Google, and many others.

    litellm doc: https://docs.litellm.ai/docs/

    Attributes:
        model_name (str):
            Model identifier. Can include provider prefix (e.g., "gpt-4", "claude-3-sonnet")
            or use provider/model format (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet").
        provider (str | None):
            Optional provider name override. If None, inferred from model_name.
            Examples: "openai", "anthropic", "google", "cohere", etc.
        base_url (str | None):
            Custom base URL for the API. If None, uses provider's default URL.
            Useful for using custom endpoints or local deployments.
        api_key (str | None):
            API key for authentication. If None, reads from environment variables.
            Environment variable names are provider-specific (e.g., OPENAI_API_KEY).
        concurrent_requests (int):
            Maximum number of concurrent API requests to execute in parallel.
            Higher values can improve throughput for batch processing but may hit rate limits
            or exhaust API quotas faster. Default is 10.
        verbose (bool):
            Whether to enable verbose logging. Default is False.
        max_model_length (int | None):
            Maximum context length for the model. If None, infers the model's default max length.
        api_max_retry (int):
            Maximum number of retries for API requests. Default is 8.
        api_retry_sleep (float):
            Initial sleep time (in seconds) between retries. Default is 1.0.
        api_retry_multiplier (float):
            Multiplier for increasing sleep time between retries. Default is 2.0.
        timeout (float):
            Request timeout in seconds. Default is None (no timeout).
        generation_parameters (GenerationParameters, optional, defaults to empty GenerationParameters):
            Configuration parameters that control text generation behavior, including
            temperature, top_p, max_new_tokens, etc.
        system_prompt (str | None, optional, defaults to None): Optional system prompt to be used with chat models.
            This prompt sets the behavior and context for the model during evaluation.
        cache_dir (str, optional, defaults to "~/.cache/huggingface/lighteval"): Directory to cache the model.
        openrouter_provider_order (list[str] | None, optional, defaults to None): For OpenRouter provider, restricts
            requests to ONLY the specified providers (e.g., ["parasail"]). This uses OpenRouter's "only" field to
            prevent fallback to other providers. Only used when provider is "openrouter" or model starts with "openrouter/".


    Example:
        ```python
        config = LiteLLMModelConfig(
            model_name="gpt-4",
            provider="openai",
            base_url="https://api.openai.com/v1",
            concurrent_requests=5,
            generation_parameters=GenerationParameters(
                temperature=0.7,
                max_new_tokens=100
            )
        )
        ```
    """

    model_name: str
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    concurrent_requests: int = 10
    verbose: bool = False
    max_model_length: int | None = None

    api_max_retry: int = 8
    api_retry_sleep: float = 1.0
    api_retry_multiplier: float = 2.0
    timeout: float | None = None

    openrouter_provider_order: list[str] | None = None


@requires("litellm")
class LiteLLMClient(LightevalModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, config: LiteLLMModelConfig) -> None:
        """IMPORTANT: Your API keys should be set in the environment variables.
        If a base_url is not set, it will default to the public API.
        """
        self.config = config
        self.model = config.model_name
        self.provider = config.provider or config.model_name.split("/")[0]
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.generation_parameters = config.generation_parameters
        self.concurrent_requests = config.concurrent_requests
        self._max_length = config.max_model_length

        self.API_MAX_RETRY = config.api_max_retry
        self.API_RETRY_SLEEP = config.api_retry_sleep
        self.API_RETRY_MULTIPLIER = config.api_retry_multiplier
        self.timeout = config.timeout

        self.openrouter_provider_order = config.openrouter_provider_order

        self._tokenizer = encode
        self.pairwise_tokenization = False
        litellm.drop_params = True
        litellm.verbose = config.verbose
        self.prompt_manager = PromptManager(
            use_chat_template=True,
            tokenizer=self.tokenizer,
            system_prompt=config.system_prompt,
        )

        # Initialize cache for tokenization and predictions
        self._cache = SampleCache(config)

    def _prepare_stop_sequence(self, stop_sequence):
        """Prepare and validate stop sequence."""
        if self.provider == "anthropic":
            # Filter out whitespace-only stop sequences
            if stop_sequence:
                stop_sequence = [s for s in stop_sequence if s and s.strip()]
        return stop_sequence

    def _prepare_max_new_tokens(self, max_new_tokens) -> int | None:
        """Calculate completion tokens based on max_new_tokens."""
        if not max_new_tokens or max_new_tokens <= 0:
            return None

        if supports_reasoning(self.model):
            # We need to allow more tokens to include reasoning tokens
            max_new_tokens = min(max_new_tokens * 10, self.max_length)

            logger.warning(
                f"Reasoning model detected, increasing max_new_tokens to {max_new_tokens} to allow for reasoning tokens",
            )

        return max_new_tokens

    def _prepare_response_format(self, response_format):
        """Prepare response format for API call."""
        if response_format is not None and BaseModel is not None:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                return response_format
        elif response_format == "json":
            return {"type": "json_object"}
        return {"type": "text"}

    def _check_finish_reason(
        self,
        choices,
        max_new_tokens: int = None,
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
                    f"max_new_tokens={max_new_tokens}, max_model_length={self.max_length}. "
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
                native_info = (
                    f" (native: {native_finish_reason})" if native_finish_reason else ""
                )
                logger.info(
                    f"Response {i + 1} finished with reason: {finish_reason}{native_info}"
                )

    def _classify_error(self, error: Exception, attempt: int) -> dict:
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

    def __call_api(
        self, prompt, return_logits, max_new_tokens, num_samples, stop_sequence
    ):  # noqa: C901
        """Make API call with retries, supporting Pydantic models in response_format and comprehensive error handling."""
        response = LitellmModelResponse()
        stop_sequence = self._prepare_stop_sequence(stop_sequence)
        max_new_tokens = self._prepare_max_new_tokens(max_new_tokens)
        response_format = self._prepare_response_format(
            self.generation_parameters.response_format
        )

        if return_logits and not self.provider == "openai":
            logger.warning(
                "Returning logits is not supported for this provider, ignoring."
            )

        # Prepare kwargs for completion call
        kwargs = {
            "model": self.model,
            "messages": prompt,
            "response_format": response_format,
            "max_tokens": max_new_tokens,
            "logprobs": return_logits if self.provider == "openai" else None,
            "stop": stop_sequence,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "n": num_samples,
            "caching": True,
            "timeout": self.timeout,
        }

        if "o1" in self.model:
            logger.warning(
                "O1 models do not support temperature, top_p, stop sequence. Disabling."
            )
        else:
            kwargs.update(self.generation_parameters.to_litellm_dict())

        if kwargs.get("max_completion_tokens", None) is None:
            kwargs["max_completion_tokens"] = max_new_tokens

        if self.openrouter_provider_order and (
            self.provider == "openrouter" or self.model.startswith("openrouter/")
        ):
            kwargs["extra_body"] = {
                "provider": {
                    "only": list(
                        self.openrouter_provider_order
                    )  # Copy list to avoid mutation issues
                }
            }

        for attempt in range(self.API_MAX_RETRY):
            try:
                response = litellm.completion(**kwargs)

                # Handle empty response with a single retry
                if not response.choices or not response.choices[0].message.content:
                    logger.info("Response is empty, retrying without caching")
                    retry_kwargs = {**kwargs, "caching": False}
                    response = litellm.completion(**retry_kwargs)

                    # Check again after retry
                    if not response.choices or not response.choices[0].message.content:
                        logger.warning(
                            "Response still empty after retry without caching"
                        )
                        return LitellmModelResponse()

                self._check_finish_reason(
                    response.choices, max_new_tokens=max_new_tokens
                )

                return response

            except Exception as e:
                error_action = self._classify_error(e, attempt)

                if error_action["should_retry"]:
                    wait_time = min(
                        64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt)
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
                    return LitellmModelResponse()

        logger.error(
            f"All {self.API_MAX_RETRY} retry attempts exhausted. Returning empty response."
        )
        return LitellmModelResponse()

    def __call_api_parallel(
        self,
        prompts,
        return_logits: bool | list[bool],
        max_new_tokens: int | list[int] | None,
        num_samples: int | list[int],
        stop_sequence: list[str] | None = None,
    ):
        results = []

        return_logitss = (
            [return_logits for _ in prompts]
            if not isinstance(return_logits, list)
            else return_logits
        )
        max_new_tokenss = (
            [max_new_tokens for _ in prompts]
            if not isinstance(max_new_tokens, list)
            else max_new_tokens
        )
        num_sampless = (
            [num_samples for _ in prompts]
            if not isinstance(num_samples, list)
            else num_samples
        )
        stop_sequencess = [stop_sequence for _ in prompts]
        assert (
            len(prompts)
            == len(return_logitss)
            == len(max_new_tokenss)
            == len(num_sampless)
            == len(stop_sequencess)
        ), f"Length of prompts, return_logitss, max_new_tokenss, num_sampless, stop_sequences, system_prompts should be the same but are {len(prompts)}, {len(return_logitss)}, {len(max_new_tokenss)}, {len(num_sampless)}, {len(stop_sequencess)}"

        with ThreadPoolExecutor(self.concurrent_requests) as executor:
            for entry in tqdm(
                executor.map(
                    self.__call_api,
                    prompts,
                    return_logitss,
                    max_new_tokenss,
                    num_sampless,
                    stop_sequencess,
                ),
                total=len(prompts),
            ):
                results.append(entry)

        if None in results:
            raise ValueError(
                "Some entries are not annotated due to errors in annotate_p, please inspect and retry."
            )

        return results

    def estimate_context_length(self) -> int:
        def fallback():
            logger.warning(
                "Failed to fetch model endpoint info from OpenRouter, returning default max length."
            )
            return self._DEFAULT_MAX_LENGTH

        # If the model is used through openrouter, the actual model name comes after the prefix
        model_name = self.model.removeprefix("openrouter/")
        endpoint_info_response = requests.get(
            f"https://openrouter.ai/api/v1/models/{model_name}/endpoints",
            headers={},
        )
        if endpoint_info_response.ok:
            try:
                endpoint_info = endpoint_info_response.json()
                context_lengths = {
                    endpoint["provider_name"]: endpoint["context_length"]
                    for endpoint in endpoint_info["data"]["endpoints"]
                }

                if self.provider in context_lengths:
                    return context_lengths[self.provider]

                min_length = min(context_lengths.values())
                logger.warning(
                    f"Estimating model context length as the minimum context length from available OpenRouter providers: {min_length}"
                )
                return min_length
            except (KeyError, TypeError, ValueError, JSONDecodeError):
                return fallback()

        return fallback()

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            docs (list[Doc]): List of documents containing the context for generation.

        Returns:
            list[ModelResponse]: list of generated responses.
        """
        dataset = GenerativeTaskDataset(
            requests=docs, num_dataset_splits=self.DATASET_SPLITS
        )
        results = []

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            contexts = [self.prompt_manager.prepare_prompt_api(doc) for doc in dataset]
            max_new_tokens = split[0].generation_size  # could be none
            return_logits = split[0].use_logits
            num_samples = split[0].num_samples
            stop_sequence = split[0].stop_sequences

            if num_samples > 1 and self.generation_parameters.temperature == 0:
                raise ValueError(
                    "num_samples > 1 is not supported with temperature=0, please set temperature > 0 or use non sampling metrics."
                )

            responses = self.__call_api_parallel(
                contexts, return_logits, max_new_tokens, num_samples, stop_sequence
            )

            for response, context in zip(responses, contexts):
                result: list[str] = [
                    choice.message.content for choice in response.choices
                ]
                reasonings: list[str | None] = [
                    getattr(choice.message, "reasoning_content", None)
                    for choice in response.choices
                ]

                cur_response = ModelResponse(
                    # In empty responses, the model should return an empty string instead of None
                    text=result if result[0] else [""],
                    reasonings=reasonings,
                    input=context,
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        if self._max_length is not None:
            return self._max_length

        try:
            max_tokens = get_max_tokens(self.model)
        except Exception:
            logger.error(
                f"Unable to get the maximum sequence length for model {self.model} from litellm. Fetching information from OpenRouter instead."
            )
            max_tokens = self.estimate_context_length()

        # Avoid future requests
        self._max_length = max_tokens

        return max_tokens

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.
        """
        raise NotImplementedError

    @cached(SamplingMethod.PERPLEXITY)
    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        raise NotImplementedError
