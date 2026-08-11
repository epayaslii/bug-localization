import os
import sys
import json
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI
from method.base import BugLocalizationMethod
from method.models import OpenAILocalizerResponse, ReasoningLocalizerResponse
from method.prompt import PromptGenerator
from dataset.utils import get_logger

try:
    from method.rag_localizer import RAGLocalizer
    RAG_IMPORT_ERROR = None
except Exception as e:
    RAGLocalizer = None
    RAG_IMPORT_ERROR = e
from pydantic import BaseModel
from dataset.utils import get_token_count
from method.utils import generate_json_schema, create_empty_localization_response, fetch_file_contents_from_github


logger = get_logger(__name__)

class OpenRouterLocalizer(BugLocalizationMethod):
    def __init__(self, model="gpt-oss-20b", max_tokens=4096, temperature=0.7):
        super().__init__()
        load_dotenv()

        self.model_mapping = {
            # free
            "gpt-oss-20b": "openai/gpt-oss-20b:free",
            # qwen3-coder's free OpenRouter tier was retired; this now routes to the paid slug.
            "qwen-coder": "qwen/qwen3-coder",
            # paid -- small/cheap
            "gpt-4o-mini": "openai/gpt-4o-mini",
            "gemini-flash": "google/gemini-flash-1.5",
            "haiku": "anthropic/claude-haiku-3-5",
        }

        self.default_model = "openai/gpt-oss-20b:free"
        
        self.model = self._validate_and_set_model(model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
        
        self.prompt_generator = PromptGenerator()
        self._rag_localizer = None
        
        self._api_call_count = 0
        self._total_tokens_used = 0
        self._successful_requests = 0
        self._failed_requests = 0
        
        logger.info(f"OpenRouterLocalizer initialized with model: {self.model} -> {self.get_model_id()}")
        logger.info(f"Initialization parameters: max_tokens={self.max_tokens}, temperature={self.temperature}")
        logger.debug(f"Available models: {len(self.model_mapping)} total")
    
    def _validate_and_set_model(self, model: str) -> str:
        if not model or not isinstance(model, str):
            logger.warning(f"Invalid model specification: {model}, using default: {self.default_model}")
            return self.default_model
        
        model = model.strip()
        
        if model in self.model_mapping:
            logger.info(f"Using friendly model name: {model} -> {self.model_mapping[model]}")
            return model
        
        logger.warning(f"Invalid model specification: {model}, using default: {self.default_model}")
        return self.default_model
        
    def get_model_id(self) -> str:
        """Get the OpenRouter model ID for API calls."""
        return self.model_mapping.get(self.model, self.model)
    
    
    def _make_api_request(self, prompt: str, structured: bool = False, response_format=None) -> str:
        try:
            model_id = self.get_model_id()
            
            request_params = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "extra_headers": {
                    "HTTP-Referer": "https://github.com/bug-localization-tool",
                    "X-Title": "Bug Localization Tool"
                }
            }
            
            if structured and response_format:
                request_params["response_format"] = response_format
            
            logger.info(f"Making API request to model: {model_id}")
            
            self._api_call_count += 1
            
            completion = self.client.chat.completions.create(**request_params)
            
            self._successful_requests += 1
            
            response_content = completion.choices[0].message.content
            
            if not response_content:
                logger.warning("API returned empty response content")
                return ""
            
            logger.info(f"API request successful, response length: {len(response_content)} characters")
            return response_content
            
        except Exception as e:
            self._failed_requests += 1
            
            from openai import APIError, RateLimitError, AuthenticationError, APIConnectionError
            
            if isinstance(e, AuthenticationError):
                logger.error("OpenRouter API authentication failed - check your API key")
                raise ValueError("Invalid OpenRouter API key") from e
            elif isinstance(e, RateLimitError):
                logger.error("OpenRouter API rate limit exceeded")
                raise ValueError("Rate limit exceeded - please try again later") from e
            elif isinstance(e, APIConnectionError):
                logger.error("Failed to connect to OpenRouter API")
                raise ValueError("Network connection error - check your internet connection") from e
            elif isinstance(e, APIError):
                error_msg = str(e)
                if "404" in error_msg and "No endpoints found" in error_msg:
                    logger.error(f"Model not available on OpenRouter: {model_id}")
                    raise ValueError(f"Model '{model_id}' is not currently available on OpenRouter. "
                                   f"Try a different model or check OpenRouter's model availability.") from e
                else:
                    logger.error(f"OpenRouter API error: {e}")
                    raise ValueError(f"API error: {e}") from e
            else:
                logger.error(f"Unexpected error during API request: {e}")
                raise ValueError(f"Unexpected error: {e}") from e

    
    def invoke_structured(self, prompt: str, text_format) -> OpenAILocalizerResponse:
        logger.info(f"Starting structured response generation with model: {self.model}")
        logger.debug(f"Response format: {text_format.__name__}")
        
        try:
            response_format = generate_json_schema(text_format)
            logger.debug(f"Generated JSON schema: {response_format}")
            
            response_content = self._make_api_request(
                prompt, 
                structured=True, 
                response_format=response_format
            )
            
            if not response_content:
                logger.warning("Structured API request returned empty content")
                return create_empty_localization_response(text_format)
            
            try:
                response_data = json.loads(response_content)
                logger.info("Successfully parsed structured JSON response")
                return text_format(**response_data)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from structured response: {e}")
                raise
            
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
        logger.info(f"Starting RAG file selection with {len(code_files)} files")

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
        logger.info(f"Repository: {bug.repo}")
        logger.info(f"Total code files: {len(bug.code_files) if bug.code_files else 0}")
        
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
                
                logger.info(f"RAG selection chose {len(selected_files)} files")
                logger.debug(f"Selected files: {selected_files[:10]}...")
                
                selected_prompt = self.prompt_generator.generate_openai_prompt(bug, selected_files)
                selected_prompt_tokens = get_token_count(selected_prompt, model="gpt-4o")
                
                logger.info(f"Selected files prompt token count: {selected_prompt_tokens}")
                
                if selected_prompt_tokens > available_tokens:
                    logger.warning(f"Selected files prompt still too large ({selected_prompt_tokens} tokens)")
                    return None
                
                response = self.invoke_structured(selected_prompt, OpenAILocalizerResponse)
                logger.info(f"RAG processing successful, found {len(response.candidate_files)} candidates")
                return response
                
            except Exception as e:
                logger.error(f"RAG processing failed: {e}")
                return self._minimal_context_fallback(bug, available_tokens)
        else:
            prompt = self.prompt_generator.generate_openai_prompt(bug)
            return self.invoke_structured(prompt, OpenAILocalizerResponse)

    def localize_with_reasoning(self, bug) -> ReasoningLocalizerResponse:
        """Like localize(), but asks the model to produce per-file reasoning before its
        final ranking (see PromptGenerator.generate_reasoning_prompt). Intended to run on
        an already BM25-narrowed bug.code_files, not the full repo -- reasoning about every
        candidate costs more output tokens than a direct pick.
        """
        logger.info(f"Starting reasoning-augmented localization for instance: {bug.instance_id}")
        logger.info(f"Candidate files: {len(bug.code_files) if bug.code_files else 0}")

        if not bug.code_files:
            logger.warning("No code files provided for localization. Check reading the code files.")
            return ReasoningLocalizerResponse(file_reasoning=[], candidate_files=[])

        prompt = self.prompt_generator.generate_reasoning_prompt(bug)
        return self.invoke_structured(prompt, ReasoningLocalizerResponse)
