from g4f.client import Client
from g4f.Provider import ChatgptAi,You,Liaobots,Bing,ChatForAi,Chatgpt4Online,ChatgptNext,ChatgptX,Gemini,GeminiPro,GptTalkRu,FlowGpt,Koala
from g4f import get_model_and_provider
import nest_asyncio
import os
from typing import Any, Generator, List, Optional

import openai
from injector import inject
from openai import AzureOpenAI, OpenAI

from taskweaver.llm.util import ChatMessageType, format_chat_message

from .base import CompletionService, EmbeddingService, LLMServiceConfig

DEFAULT_STOP_TOKEN: List[str] = ["<EOS>"]


# nest_asyncio.apply()


class Gpt4freeServiceConfig(LLMServiceConfig):
    def _configure(self) -> None:
        # shared common config
        self._set_name('gpt4free')

        self.api_type = self._get_str(
            "api_type",
            None,
        )
        shared_api_base = self.llm_module_config.api_base
        self.api_base = self._get_str(
            "api_base",
            shared_api_base,
        )
        shared_api_key = self.llm_module_config.api_key
        self.api_key = self._get_str(
            "api_key",
            shared_api_key,
        )

        shared_model = self.llm_module_config.model
        self.model = self._get_str(
            "model",
            shared_model,
        )

        provider = self._get_str(
            "provider",
            None,
            required=False
        )
        _,self.provider = get_model_and_provider(self.model,provider,stream=True)

        shared_backup_model = self.llm_module_config.backup_model
        self.backup_model = self._get_str(
            "backup_model",
            shared_backup_model if shared_backup_model is not None else self.model,
        )
        shared_embedding_model = self.llm_module_config.embedding_model
        self.embedding_model = self._get_str(
            "embedding_model",
            shared_embedding_model if shared_embedding_model is not None else "text-embedding-ada-002",
        )

        self.response_format = self.llm_module_config.response_format

        # openai specific config
        self.api_version = self._get_str("api_version", "2023-12-01-preview")
        self.api_auth_type = self._get_enum(
            "api_auth_type",
            ["openai", "azure", "azure_ad"],
            "openai",
        )

        self.stop_token = self._get_list("stop_token", DEFAULT_STOP_TOKEN)
        self.temperature = self._get_float("temperature", 0)
        self.max_tokens = self._get_int("max_tokens", 1024)
        self.top_p = self._get_float("top_p", 0)
        self.frequency_penalty = self._get_float("frequency_penalty", 0)
        self.presence_penalty = self._get_float("presence_penalty", 0)
        self.seed = self._get_int("seed", 123456)


class Gpt4freeService(CompletionService, EmbeddingService):
    @inject
    def __init__(self, config: Gpt4freeServiceConfig):
        self.config = config

        api_type = self.config.api_type

        self.client: Client = Client(provider=self.config.provider)

    def chat_completion(
        self,
        messages: List[ChatMessageType],
        stream: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Generator[ChatMessageType, None, None]:
        engine = self.config.model
        self.config.backup_model

        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        top_p = top_p if top_p is not None else self.config.top_p
        stop = stop if stop is not None else self.config.stop_token
        seed = self.config.seed

        try:
            tools_kwargs = {}
            if "tools" in kwargs and "tool_choice" in kwargs:
                tools_kwargs["tools"] = kwargs["tools"]
                tools_kwargs["tool_choice"] = kwargs["tool_choice"]
            if "response_format" in kwargs:
                response_format = kwargs["response_format"]
            elif self.config.response_format == "json_object":
                response_format = {"type": "json_object"}
            else:
                response_format = None

            res: Any = self.client.chat.completions.create(
                model=engine,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                stop=stop,
                stream=stream,
                seed=seed,
                response_format=response_format,
                **tools_kwargs,
            )
            if stream:
                role: Any = None
                for stream_res in res:
                    if not stream_res.choices:
                        continue
                    delta = stream_res.choices[0].delta
                    if delta is None:
                        continue

                    # role = delta.role if delta.role is not None else role
                    role = "assistant"
                    content = delta.content if delta.content is not None else ""
                    if content is None:
                        continue
                    yield format_chat_message(role, content)
            else:
                oai_response = res.choices[0].message
                if oai_response is None:
                    raise Exception("OpenAI API returned an empty response")
                response: ChatMessageType = format_chat_message(
                    role=oai_response.role if oai_response.role is not None else "assistant",
                    message=oai_response.content if oai_response.content is not None else "",
                )
                if oai_response.tool_calls is not None:
                    import json

                    response["role"] = "function"
                    response["content"] = json.dumps(
                        [
                            {
                                "name": t.function.name,
                                "arguments": json.loads(t.function.arguments),
                            }
                            for t in oai_response.tool_calls
                        ],
                    )
                yield response

        except openai.APITimeoutError as e:
            # Handle timeout error, e.g. retry or log
            raise Exception(f"OpenAI API request timed out: {e}")
        except openai.APIConnectionError as e:
            # Handle connection error, e.g. check network or log
            raise Exception(f"OpenAI API request failed to connect: {e}")
        except openai.BadRequestError as e:
            # Handle invalid request error, e.g. validate parameters or log
            raise Exception(f"OpenAI API request was invalid: {e}")
        except openai.AuthenticationError as e:
            # Handle authentication error, e.g. check credentials or log
            raise Exception(f"OpenAI API request was not authorized: {e}")
        except openai.PermissionDeniedError as e:
            # Handle permission error, e.g. check scope or log
            raise Exception(f"OpenAI API request was not permitted: {e}")
        except openai.RateLimitError as e:
            # Handle rate limit error, e.g. wait or log
            raise Exception(f"OpenAI API request exceeded rate limit: {e}")
        except openai.APIError as e:
            # Handle API error, e.g. retry or log
            raise Exception(f"OpenAI API returned an API Error: {e}")

    def get_embeddings(self, strings: List[str]) -> List[List[float]]:
        embedding_results = self.client.embeddings.create(
            input=strings,
            model=self.config.embedding_model,
        ).data
        return [r.embedding for r in embedding_results]
