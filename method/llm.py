import requests
import os 
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from typing import Type, Any
from pydantic import BaseModel
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.utils import get_logger
load_dotenv()

logger = get_logger(__name__)

class LLMClientGenerator:
    def __init__(self, api_key: str = None, use_openrouter: bool = False, openrouter_api_key: str = None):
        self.use_openrouter = use_openrouter
        self.model_mapping = {}

        if self.use_openrouter:
            self.api_key = os.getenv("OPENROUTER_API_KEY", None)
            
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )
            
            self.llm_web_service_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
                "X-Title": os.getenv("OPENROUTER_SITE_NAME", "Bug Localization")
            }
        else:
            if not api_key:
                self.api_key = os.getenv("OPENAI_API_KEY", None)
            else:
                self.api_key = api_key
            
            self.llm_web_service_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            self.client = OpenAI(api_key=self.api_key)

    def _get_model_name(self, model_type: str) -> str:
        if self.use_openrouter and model_type in self.model_mapping:
            return self.model_mapping[model_type]
        return model_type
    
    def _pydantic_to_json_schema(self, model_class: Type[BaseModel]) -> dict:
        schema = model_class.model_json_schema()
        
        return {
            "name": model_class.__name__.lower(),
            "strict": True,
            "schema": schema
        }

    def invoke(self, prompt, model_type="gpt-5-nano") -> str:
        full_model_name = model_type
        
        if self.use_openrouter:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
                    "X-Title": os.getenv("OPENROUTER_SITE_NAME", "Bug Localization"),
                },
                model=full_model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
        else:
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": full_model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            response = requests.post(url, json=payload, headers=self.llm_web_service_headers)
            if response.status_code == 200:
                llm_response = response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")
            return llm_response
    

    def invoke_structured(self, prompt, text_format, model="gpt-4o") -> Any:
        full_model_name = self._get_model_name(model)
        
        if self.use_openrouter:
            json_schema = self._pydantic_to_json_schema(text_format)
            schema_description = json.dumps(json_schema, indent=2)
            structured_prompt = f"""{prompt}

            Please respond with a valid JSON object that matches this exact schema:
            {schema_description}

            Respond ONLY with the JSON object, no additional text."""
            
            try:
                completion = self.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
                        "X-Title": os.getenv("OPENROUTER_SITE_NAME", "Bug Localization"),
                    },
                    model=full_model_name,
                    messages=[
                        {"role": "user", "content": structured_prompt}
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": json_schema
                    }
                )
                
                response_content = completion.choices[0].message.content
                
                logger.debug(f"Raw response content: {response_content}")
                
                response_content = response_content.strip()
                
                try:
                    response_data = json.loads(response_content)
                    logger.debug(f"Parsed JSON data: {response_data}")
                    return text_format(**response_data)
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON parsing error: {json_err}")
                    logger.error(f"Response content that failed to parse: {repr(response_content)}")
                    json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                    if json_match:
                        try:
                            clean_json = json_match.group(0)
                            response_data = json.loads(clean_json)
                            logger.info(f"Successfully parsed JSON after cleaning: {response_data}")
                            return text_format(**response_data)
                        except json.JSONDecodeError:
                            logger.error("Failed to parse even after cleaning")
                    raise json_err
                except Exception as pydantic_err:
                    logger.error(f"Pydantic validation error: {pydantic_err}")
                    logger.error(f"Data that failed validation: {response_data}")
                    raise pydantic_err
                
            except Exception as e:
                logger.error(f"Structured output failed for {full_model_name}: {e}")
                raise e
        else:
            response = self.client.responses.parse(
                model=full_model_name,
                input=[{"role":"user", "content": prompt}],
                text_format=text_format
            )
            return response.output_parsed
    


