from dataset.utils import get_logger
import requests
import base64
import os
    
logger = get_logger(__name__)

def generate_json_schema(pydantic_model):
        try:
            model_schema = pydantic_model.model_json_schema()

            schema_name = pydantic_model.__name__.lower().replace('response', '_response')

            if 'required' not in model_schema:
                model_schema['required'] = list(model_schema.get('properties', {}).keys())

            model_schema['additionalProperties'] = False

            # OpenAI-style strict mode requires every object in the schema -- including
            # nested $defs from models with object-typed fields (e.g. a list of judgment
            # objects) -- to set additionalProperties: false and list every property as
            # required, not just the top-level schema.
            for def_schema in model_schema.get('$defs', {}).values():
                if def_schema.get('type') == 'object':
                    def_schema['additionalProperties'] = False
                    def_schema['required'] = list(def_schema.get('properties', {}).keys())

            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": model_schema
                }
            }
        except Exception as e:
            logger.warning(f"Failed to generate JSON schema from Pydantic model: {e}")
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "bug_localization_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "candidate_files": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["candidate_files"],
                        "additionalProperties": False
                    }
                }
            }


def create_empty_localization_response(text_format):
        try:
            if hasattr(text_format, 'model_fields'):
                empty_data = {}
                for field_name, field_info in text_format.model_fields.items():
                    if field_info.annotation == list or (hasattr(field_info.annotation, '__origin__') and field_info.annotation.__origin__ == list):
                        empty_data[field_name] = []
                    elif field_info.annotation == str:
                        empty_data[field_name] = ""
                    else:
                        empty_data[field_name] = None
                return text_format(**empty_data)
            else:
                return text_format(candidate_files=[])
        except Exception as e:
            logger.warning(f"Failed to create empty localization response: {e}")
            return text_format(candidate_files=[])

def fetch_file_contents_from_github(bug):
    from dataset.repo_cache import is_repo_cached, get_file_content_local

    file_contents = {}
    successful_fetches = 0
    total_fetches = 0

    github_token = os.getenv("GITHUB_TOKEN")
    use_local_cache = is_repo_cached(bug.repo)

    for file_path in bug.code_files:
        if ".git" in file_path or file_path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf')):
            continue

        total_fetches += 1

        if use_local_cache:
            try:
                content = get_file_content_local(bug.repo, bug.base_commit, file_path)
                file_contents[file_path] = content
                successful_fetches += 1
                continue
            except Exception as e:
                logger.warning(f"Local cache read failed for {file_path}@{bug.base_commit}, falling back to GitHub API: {e}")

        try:
            url = f"https://api.github.com/repos/{bug.repo}/contents/{file_path}?ref={bug.base_commit}"
            headers = {"Accept": "application/vnd.github.v3+json"}
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            content = base64.b64decode(response.json()["content"]).decode('utf-8')
            file_contents[file_path] = content
            successful_fetches += 1

            if successful_fetches % 10 == 0:
                logger.info(f"Fetched {successful_fetches} files from GitHub...")
                logger.info(f"Total fetches: {total_fetches}")
        except Exception as e:
            logger.warning(f"Failed to fetch {file_path} from GitHub: {e}")
            logger.info(f"Total fetches: {total_fetches}")
            continue

    logger.info(f"Total fetches: {total_fetches}")
    logger.info(f"Successful fetches: {successful_fetches}")
    return file_contents

