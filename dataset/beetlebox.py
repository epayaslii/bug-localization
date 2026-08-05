import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.base import BugLocalizationDataset
import logging
from datasets import load_dataset, load_from_disk
from dataset.models import BugInstance
from dataset.utils import get_code_files, calculate_dataset_token_stats, is_code_file, filter_code_paths
from dotenv import load_dotenv
from datetime import datetime
import re
import random

logger = logging.getLogger(__name__)

class BeetleBox(BugLocalizationDataset):
    """
    BeetleBox dataset loader for multi-language bug localization.
    
    The BeetleBox dataset contains 26,321 bugs from 29 projects across 5 programming languages:
    Java, Python, C++, JavaScript, and Go.
    
    Dataset fields:
    - status: The current status of the bug report
    - repo_name: The name of the repository
    - repo_url: The URL of the repository
    - issue_id: The unique identifier for the issue
    - updated_files: A list of files that were updated during the bug fix
    - title: The title of the bug report
    - body: The main content or description of the bug report
    - issue_url: The URL of the issue
    - pull_url: The URL of the pull request associated with the bug fix
    - before_fix_sha: The SHA hash of the commit before the bug fix
    - after_fix_sha: The SHA hash of the commit after the bug fix
    - report_datetime: The date and time when the bug was reported
    - language: The programming language of the project
    - commit_datetime: The date and time when the bug fix was committed
    """

    def __init__(self, split='train', language_filter=None, repo_filter=None):
        """
        Initialize BeetleBox dataset.
        
        Args:
            split: Dataset split to load ('train', 'test', 'validation')
            language_filter: Optional language filter (e.g., 'python', 'java', 'cpp', 'javascript', 'go')
            repo_filter: Optional repository name filter
        """
        super().__init__()
        load_dotenv()
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            logger.warning("⚠️ GITHUB_TOKEN not found in environment. You may hit rate limits.")

        logger.info("Initializing BeetleBox dataset")
        self.dataset_name = "BeetleBox"
        self.split = split
        self.language_filter = language_filter
        self.repo_filter = repo_filter
        self._bug_instances = []
        self.data = None
        
        # Set extensions based on language filter
        self.extensions = self._get_extensions()
        
        self.load_data()

    def _get_extensions(self):
        """Get file extensions based on language filter."""
        extension_map = {
            'python': ('.py',),
            'java': ('.java',),
            'cpp': ('.cpp', '.cc', '.cxx', '.h', '.hpp'),
            'javascript': ('.js', '.jsx', '.ts', '.tsx'),
            'go': ('.go',)
        }
        
        if self.language_filter:
            return extension_map.get(self.language_filter.lower(), ('.py', '.java', '.cpp', '.js', '.go'))
        else:
            all_extensions = []
            for exts in extension_map.values():
                all_extensions.extend(exts)
            return tuple(all_extensions)

    def load_data(self):
        """Load BeetleBox dataset from Hugging Face, or from local disk if BEETLEBOX_LOCAL_PATH is set."""
        local_path = os.environ.get("BEETLEBOX_LOCAL_PATH")
        if local_path:
            logger.info(f"Loading BeetleBox dataset from local disk: {local_path}")
            self.data = load_from_disk(local_path)
            logger.info(f"BeetleBox dataset loaded successfully. Total instances: {len(self.data)}")

            if self.data and len(self.data) > 0:
                sample = self.data[0]
                logger.info(f"Dataset fields: {list(sample.keys())}")
            return

        logger.info(f"Loading BeetleBox dataset from bug-localization/BeetleBox (split: {self.split})...")
        try:
            # Load the dataset from Hugging Face
            self.data = load_dataset('bug-localization/BeetleBox', split=self.split)
            logger.info(f"BeetleBox dataset loaded successfully. Total instances: {len(self.data)}")

            # Log dataset fields for inspection
            if self.data and len(self.data) > 0:
                sample = self.data[0]
                logger.info(f"Dataset fields: {list(sample.keys())}")

        except Exception as e:
            logger.error(f"Failed to load BeetleBox dataset: {e}")
            raise

    def inspect_dataset_fields(self):
        """Inspect and log the fields available in the dataset."""
        if self.data is None:
            logger.warning("Data not loaded, attempting to load...")
            self.load_data()
        
        if len(self.data) > 0:
            sample = self.data[0]
            logger.info("BeetleBox Dataset Fields:")
            for field, value in sample.items():
                logger.info(f"  {field}: {type(value).__name__} - {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
            return sample
        else:
            logger.warning("Dataset is empty")
            return None

    def get_bug_instances(self, sample_size=None, random_sample=False, random_seed=None):
        if self.data is None:
            logger.warning("Data not loaded, attempting to load...")
            self.load_data()
                
        if self._bug_instances:
            logger.debug(f"Returning cached bug instances: {len(self._bug_instances)}")
            if sample_size is not None and sample_size < len(self._bug_instances):
                if random_sample:
                    if random_seed is not None:
                        random.seed(random_seed)
                    return random.sample(self._bug_instances, sample_size)
                else:
                    return self._bug_instances[:sample_size]
            return self._bug_instances

        logger.info("Processing bug instances...")
        
        if hasattr(self.data, 'to_dict'):
            bug_instances = [dict(zip(self.data.features.keys(), instance)) 
                                  for instance in zip(*self.data.to_dict().values())]
        else:
            bug_instances = self.data

        if sample_size is not None and sample_size < len(bug_instances):
            if random_sample:
                if random_seed is not None:
                    random.seed(random_seed)
                bug_instances = random.sample(bug_instances, sample_size)
                logger.info(f"Randomly sampled {len(bug_instances)} instances from {len(self.data)} total")
            else:
                bug_instances = bug_instances[:sample_size]
                logger.info(f"Taking first {len(bug_instances)} instances from {len(self.data)} total")
        else:
            if sample_size is None:
                sample_size = len(bug_instances)
            logger.info(f"Processing all {len(bug_instances)} bug instances...")
        
        processed_count = 0
        for i, bug in enumerate(bug_instances):
            if i % 100 == 0: 
                logger.debug(f"Processed {i}/{len(bug_instances)} bug instances")
            
            if self.language_filter and bug.get('language', '').lower() != self.language_filter.lower():
                continue
                
            if self.repo_filter and self.repo_filter not in bug.get('repo_name', ''):
                continue
            
            try:
                repo_name = bug.get('repo_name', '')
                repo_url = bug.get('repo_url', '')
                
                title = bug.get('title', '')
                body = bug.get('body', '')
                bug_report = f"Title: {title}\n\nDescription:\n{body}"
                
                updated_files = bug.get('updated_files', [])
                if isinstance(updated_files, str):
                    try:
                        updated_files = eval(updated_files) if updated_files else []
                    except:
                        updated_files = [updated_files] if updated_files else []
                # Keep only code-file ground truths; skip instance if none remain
                updated_files = [p for p in updated_files if isinstance(p, str) and p.endswith(self.extensions)]
                if not updated_files:
                    continue
                
                before_commit = bug.get('before_fix_sha', '')
                after_commit = bug.get('after_fix_sha', '')
                
                code_files = []
                if repo_name and before_commit:
                    try:
                        code_files = get_code_files(repo_name, before_commit, self.extensions, self.token)
                    except Exception as e:
                        logger.warning(f"Failed to get code files for {repo_name}@{before_commit}: {e}")
                        code_files = []
                
                bug_instance = BugInstance(
                    instance_id=str(bug.get('issue_id', i)),
                    repo=repo_name,
                    base_commit=before_commit,
                    patch=f"Before: {before_commit}\nAfter: {after_commit}",
                    hints_text=f"Status: {bug.get('status', '')}\nLanguage: {bug.get('language', '')}\nIssue URL: {bug.get('issue_url', '')}\nPR URL: {bug.get('pull_url', '')}",
                    ground_truths=updated_files,
                    bug_report=bug_report,
                    code_files=code_files,
                    after_commit=after_commit or None,
                )
                
                self._bug_instances.append(bug_instance)
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to process bug instance {i}: {e}")
                continue
        
        self.repos = list(set([bug.repo for bug in self._bug_instances]))
        
        logger.info(f"Found {len(self.repos)} unique repositories")
        logger.info(f"Successfully processed {processed_count} bug instances from {len(bug_instances)} total")
        
        return self._bug_instances

    def get_token_statistics(self, model="gpt-4o", sample_size=100):
        """
        Get token count statistics for the entire dataset with efficient sampling.
        
        Args:
            model: Model name for token counting
            sample_size: Number of instances to sample for calculation (default: 1000)
        
        Returns:
            Dictionary with token statistics including estimated totals
        """
        if not self._bug_instances:
            self.get_bug_instances()
        
        stats = calculate_dataset_token_stats(self._bug_instances, model, sample_size=sample_size)
        
        logger.info(f"Dataset Token Statistics ({model}):")
        logger.info(f"  Total instances: {stats['total_instances']:,}")
        if stats['is_sampled']:
            logger.info(f"  Sample size: {stats['sample_size']:,}")
            logger.info(f"  ⚠️  Statistics calculated from sample")
        logger.info(f"  Mean tokens per instance: {stats['mean_tokens']:.2f}")
        logger.info(f"  Min tokens: {stats['min_tokens']:,}")
        logger.info(f"  Max tokens: {stats['max_tokens']:,}")
        if stats['is_sampled']:
            logger.info(f"  Estimated total tokens: {stats['estimated_total_tokens']:,}")
        else:
            logger.info(f"  Total tokens across dataset: {stats['total_tokens']:,}")
        
        return stats

    def get_language_distribution(self):
        """Get distribution of programming languages in the dataset."""
        if self.data is None:
            self.load_data()
        
        languages = {}
        for item in self.data:
            lang = item.get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
        
        logger.info("Language distribution:")
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {lang}: {count}")
        
        return languages

    def get_repository_distribution(self):
        """Get distribution of repositories in the dataset."""
        if self.data is None:
            self.load_data()
        
        repos = {}
        for item in self.data:
            repo = item.get('repo_name', 'unknown')
            repos[repo] = repos.get(repo, 0) + 1
        
        logger.info("Repository distribution:")
        for repo, count in sorted(repos.items(), key=lambda x: x[1], reverse=True)[:20]:  # Top 20
            logger.info(f"  {repo}: {count}")
        
        return repos
