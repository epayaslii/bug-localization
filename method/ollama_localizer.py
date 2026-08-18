import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from method.base import BugLocalizationMethod
from method.models import OpenAILocalizerResponse
from method.prompt import PromptGenerator
from dataset.utils import get_logger
from dataset.utils import get_token_count
from method.utils import generate_json_schema, create_empty_localization_response, fetch_file_contents_from_github

try:
    from method.rag_localizer import RAGLocalizer
    RAG_IMPORT_ERROR = None
except Exception as e:
    RAGLocalizer = None
    RAG_IMPORT_ERROR = e

logger = get_logger(__name__)


class OllamaModelNotFoundError(RuntimeError):
    """Raised (not caught by invoke_structured's broad except) when Ollama 404s with
    "model not found" -- a config mistake (wrong tag, or pulled/transferred a different
    variant than requested), not a transient runtime hiccup. Letting this one propagate and
    crash the run immediately matters here specifically: invoke_structured's normal
    exception handling silently falls back to an empty judgment per call, which for a
    misconfigured model means every instance in a run "succeeds" with 0 relevant judgments
    -- a real run was burned exactly this way on MN5 (2026-08-18, requested
    qwen2.5-coder:14b, only :7b had been pulled/transferred) before this got explicit
    handling."""


class OllamaLocalizer(BugLocalizationMethod):
    """LLM reranking against a local Ollama server instead of a cloud API -- the deployment
    path for the supervisor's target architecture, which needs to run somewhere with no
    outbound internet (e.g. MN5). Ollama exposes an OpenAI-compatible /v1/chat/completions
    endpoint (including response_format json_schema, same as OpenRouterLocalizer uses), so
    this mirrors OpenRouterLocalizer almost exactly -- only base_url/api_key/model_mapping
    differ. See docs/ollama_deployment.md for the local-vs-MN5 setup and model choice
    rationale.
    """

    def __init__(self, model="qwen2.5-coder", max_tokens=4096, temperature=0.7, host=None, num_ctx=16384):
        super().__init__()

        # Friendly names -> the actual `ollama pull`-able tag. Update this mapping (and
        # docs/ollama_deployment.md) if the recommended default model changes.
        self.model_mapping = {
            "qwen2.5-coder": "qwen2.5-coder:14b",
            "qwen2.5-coder-32b": "qwen2.5-coder:32b",
            "qwen2.5-coder-7b": "qwen2.5-coder:7b",
            "llama3.1": "llama3.1:8b",
            "deepseek-coder-v2": "deepseek-coder-v2:16b",
        }
        self.default_model = "qwen2.5-coder:14b"

        self.model = self._validate_and_set_model(model)

        self.max_tokens = max_tokens
        self.temperature = temperature
        # Ollama's OpenAI-compat endpoint defaults the runtime context window to 4096
        # tokens regardless of the model's real max (e.g. qwen2.5:latest supports 32768
        # natively per `ollama show`) -- silently truncates anything longer, which a
        # chunk-granularity relevance-feedback prompt (many candidates x several chunks
        # each) blows past easily. Passed through as `options.num_ctx` (Ollama-specific,
        # not a standard OpenAI field) via extra_body.
        self.num_ctx = num_ctx

        # OLLAMA_HOST matches Ollama's own env var convention, so a job script that already
        # sets it (e.g. to point at a non-default port) needs no extra config here.
        base_host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if not base_host.startswith("http"):
            base_host = f"http://{base_host}"
        self.base_url = f"{base_host.rstrip('/')}/v1"

        # Ollama doesn't check the key, but the OpenAI SDK requires a non-empty string.
        self.client = OpenAI(base_url=self.base_url, api_key="ollama")

        self.prompt_generator = PromptGenerator()
        self._rag_localizer = None

        self._api_call_count = 0
        self._total_tokens_used = 0
        self._successful_requests = 0
        self._failed_requests = 0

        logger.info(f"OllamaLocalizer initialized with model: {self.model} -> {self.get_model_id()} at {self.base_url}")
        logger.info(f"Initialization parameters: max_tokens={self.max_tokens}, temperature={self.temperature}")

    def _validate_and_set_model(self, model: str) -> str:
        if not model or not isinstance(model, str):
            logger.warning(f"Invalid model specification: {model}, using default: {self.default_model}")
            return self.default_model

        model = model.strip()

        if model in self.model_mapping:
            logger.info(f"Using friendly model name: {model} -> {self.model_mapping[model]}")
            return model

        # Also accept a raw Ollama tag directly (e.g. "qwen2.5-coder:32b-instruct-q4_K_M")
        # for models not in the friendly-name table above.
        if ":" in model:
            return model

        logger.warning(f"Invalid model specification: {model}, using default: {self.default_model}")
        return self.default_model

    def get_model_id(self) -> str:
        return self.model_mapping.get(self.model, self.model)

    def _make_api_request(self, prompt: str, structured: bool = False, response_format=None) -> str:
        try:
            model_id = self.get_model_id()

            request_params = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "extra_body": {"options": {"num_ctx": self.num_ctx}},
            }

            if structured and response_format:
                request_params["response_format"] = response_format

            logger.info(f"Making API request to local Ollama model: {model_id}")

            self._api_call_count += 1

            completion = self.client.chat.completions.create(**request_params)

            self._successful_requests += 1

            response_content = completion.choices[0].message.content

            if not response_content:
                logger.warning("Ollama returned empty response content")
                return ""

            logger.info(f"Ollama request successful, response length: {len(response_content)} characters")
            return response_content

        except Exception as e:
            self._failed_requests += 1

            from openai import APIConnectionError

            if isinstance(e, APIConnectionError):
                logger.error(f"Failed to connect to Ollama at {self.base_url} -- is `ollama serve` running?")
                raise ValueError(
                    f"Could not reach Ollama at {self.base_url}. Start it with `ollama serve` "
                    f"(and `ollama pull {self.get_model_id()}` if the model isn't pulled yet)."
                ) from e
            elif "not found" in str(e) and "pulling" in str(e):
                logger.critical(f"Ollama model '{self.get_model_id()}' is not pulled/available at {self.base_url} -- aborting.")
                raise OllamaModelNotFoundError(
                    f"Ollama model '{self.get_model_id()}' not found at {self.base_url}. "
                    f"Run `ollama pull {self.get_model_id()}` (and transfer it to MN5 if remote) before retrying."
                ) from e
            else:
                logger.error(f"Unexpected error during Ollama request: {e}")
                raise ValueError(f"Unexpected error: {e}") from e

    def invoke_structured(self, prompt: str, text_format) -> OpenAILocalizerResponse:
        logger.info(f"Starting structured response generation with model: {self.model}")

        try:
            response_format = generate_json_schema(text_format)

            response_content = self._make_api_request(
                prompt,
                structured=True,
                response_format=response_format
            )

            if not response_content:
                logger.warning("Structured Ollama request returned empty content")
                return create_empty_localization_response(text_format)

            try:
                response_data = json.loads(response_content)
                logger.info("Successfully parsed structured JSON response")
                return text_format(**response_data)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from structured response: {e}")
                raise

        except OllamaModelNotFoundError:
            raise  # config error, not a per-call hiccup -- let it crash the run, see class docstring

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Structured response parsing failed: {e}")
            return create_empty_localization_response(text_format)

        except Exception as e:
            logger.error(f"Structured response generation failed: {e}")
            return create_empty_localization_response(text_format)

    def total_stats(self):
        return {
            "api_call_count": self._api_call_count,
            "total_tokens_used": self._total_tokens_used,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests
        }

    def _get_rag_files(self, bug, code_files, file_contents):
        if RAGLocalizer is None:
            raise RuntimeError(f"RAG requested but rag_localizer failed to import: {RAG_IMPORT_ERROR}")

        if self._rag_localizer is None:
            self._rag_localizer = RAGLocalizer()

        self._rag_localizer.create_collection(code_files, file_contents)
        selected_files = self._rag_localizer.search_relevant_files(bug.bug_report, top_k=100)

        logger.info(f"RAG selection completed, selected {len(selected_files)} files")
        return selected_files

    def localize(self, bug, rag=False):
        logger.info(f"Starting bug localization for instance: {bug.instance_id}")

        if not bug.code_files:
            logger.warning("No code files provided for localization. Check reading the code files.")
            return OpenAILocalizerResponse(candidate_files=[])

        if rag:
            logger.info("Using RAG file selection to reduce context size")
            try:
                file_contents = fetch_file_contents_from_github(bug)
                selected_files = self._get_rag_files(bug, bug.code_files, file_contents)

                if not selected_files:
                    logger.warning("RAG selection returned no files")
                    return OpenAILocalizerResponse(candidate_files=[])

                selected_prompt = self.prompt_generator.generate_openai_prompt(bug, selected_files)
                return self.invoke_structured(selected_prompt, OpenAILocalizerResponse)

            except Exception as e:
                logger.error(f"RAG processing failed: {e}")
                return OpenAILocalizerResponse(candidate_files=[])
        else:
            prompt = self.prompt_generator.generate_openai_prompt(bug)
            return self.invoke_structured(prompt, OpenAILocalizerResponse)
