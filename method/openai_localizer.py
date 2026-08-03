import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method.base import BugLocalizationMethod
from openai import OpenAI
from dotenv import load_dotenv
import os
from method.prompt import PromptGenerator
from method.models import OpenAILocalizerResponse
from method.llm import LLMClientGenerator
from dataset.utils import get_token_count, chunk_code_files, estimate_prompt_tokens, get_logger

logger = get_logger(__name__)

class OpenAILocalizer(BugLocalizationMethod):
    def __init__(self, model="gpt-5-nano"):
        super().__init__()
        load_dotenv()
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = LLMClientGenerator(api_key=self.api_key)
        self.prompt_generator = PromptGenerator()

    def localize(self, bug, max_prompt_tokens=100000, max_chunk_tokens=50000):
        model = self.model

        # Count tokens in the bug report using the utility function
        token_count = get_token_count(bug.to_string(), model=model)

        # If the bug report is too long, summarize it
        if token_count > 4096:
            prompt = self.prompt_generator.generate_openai_report_summarizer_prompt(bug)
            bug_report = self.llm.invoke(prompt)
            # Update the bug object with the summarized report
            bug.bug_report = bug_report
        
        # Check if code files need chunking
        code_files_tokens = get_token_count("\n\n".join(bug.code_files), model)
        
        if code_files_tokens <= max_chunk_tokens:
            # No chunking needed - process normally
            prompt = self.prompt_generator.generate_openai_prompt(bug)
            
            response = self.llm.invoke_structured(
                prompt=prompt,
                text_format=OpenAILocalizerResponse,
                model=model
            )
            return response
        
        else:
            # Need chunking - split code files and process each chunk
            logger.info(f"Code files require chunking: {code_files_tokens} tokens > {max_chunk_tokens} limit")
            
            # Split code files into manageable chunks
            chunks = chunk_code_files(bug.code_files, max_chunk_tokens, model)
            
            chunk_responses = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} with {len(chunk)} files...")
                
                # Estimate total prompt tokens for this chunk
                estimated_tokens = estimate_prompt_tokens(bug, chunk, model)
                
                if estimated_tokens > max_prompt_tokens:
                    logger.warning(f"Chunk {i+1} estimated at {estimated_tokens} tokens, may exceed model limits")
                
                # Generate prompt for this chunk
                chunk_prompt = self.prompt_generator.generate_openai_prompt(bug, chunk)
                
                # Get response for this chunk
                try:
                    chunk_response = self.llm.invoke_structured(
                        prompt=chunk_prompt,
                        text_format=OpenAILocalizerResponse,
                        model=model
                    )
                    chunk_responses.append(chunk_response)
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {e}")
                    # Continue with other chunks
                    continue
            
            # If we only have one successful response, return it
            if len(chunk_responses) == 1:
                return chunk_responses[0]
            
            # If we have multiple responses, aggregate them
            elif len(chunk_responses) > 1:
                return self._aggregate_chunk_responses(bug, chunk_responses, model)
            
            # If no chunks were successfully processed, fall back to original approach
            else:
                logger.warning("All chunks failed, falling back to original approach")
                prompt = self.prompt_generator.generate_openai_prompt(bug)
                return self.llm.invoke_structured(
                    prompt=prompt,
                    text_format=OpenAILocalizerResponse,
                    model=model
                )
    
    def _aggregate_chunk_responses(self, bug, chunk_responses, model):
        """
        Aggregate responses from multiple chunks into a final response.
        
        Args:
            bug: BugInstance object
            chunk_responses: List of OpenAILocalizerResponse objects
            model: Model name
        """
        logger.info(f"Aggregating {len(chunk_responses)} chunk responses...")
        
        # Convert structured responses to text for aggregation
        response_texts = []
        for i, response in enumerate(chunk_responses):
            # Convert structured response to text representation
            response_text = f"Files analyzed: {getattr(response, 'files', 'N/A')}\n"
            response_text += f"Reasoning: {getattr(response, 'reasoning', 'N/A')}"
            response_texts.append(response_text)
        
        # Generate aggregation prompt
        aggregation_prompt = self.prompt_generator.generate_chunk_aggregation_prompt(bug, response_texts)
        
        # Get final aggregated response
        try:
            final_response = self.llm.invoke_structured(
                prompt=aggregation_prompt,
                text_format=OpenAILocalizerResponse,
                model=model
            )
            return final_response
            
        except Exception as e:
            logger.error(f"Error during aggregation: {e}")
            # Fall back to the first successful chunk response
            return chunk_responses[0]
