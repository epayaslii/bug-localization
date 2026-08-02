import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.base import BugLocalizationDataset
import logging
import re
from datasets import load_dataset
from dataset.models import BugInstance
from dataset.utils import get_code_files, calculate_dataset_token_stats, is_code_file, filter_code_paths
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class SWEBench(BugLocalizationDataset):

    def __init__(self):
        super().__init__()
        load_dotenv()
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            logger.warning("GITHUB_TOKEN not found in environment. You may hit rate limits.")

        logger.info("Initializing SWEBench dataset")
        self.dataset_name = "SWE-bench_Verified"
        self._bug_instances = []
        self.data = None
        self.load_data()
        self.extensions = (".py")
    

    def load_data(self):
        logger.info("Loading SWE-bench Verified dataset from SWE-bench/SWE-bench_Verified...")
        try:
            self.data = load_dataset('SWE-bench/SWE-bench_Verified', split='test')
            logger.info(f"SWE-bench Verified dataset loaded successfully. Total instances: {len(self.data)}")
        except Exception as e:
            logger.error(f"Failed to load SWE-bench Verified dataset: {e}")
            raise

    def get_bug_instances(self, sample_size=None, random_sample=False, random_seed=None):
        if self.data is None:
            logger.warning("Data not loaded, attempting to load...")
            self.load_data()
        
        if self._bug_instances and sample_size is None:
            logger.debug(f"Returning cached bug instances: {len(self._bug_instances)}")
            return self._bug_instances
        
        if self._bug_instances and sample_size is not None:
            if sample_size >= len(self._bug_instances):
                logger.debug(f"Sample size {sample_size} >= cached instances {len(self._bug_instances)}, returning all cached")
                return self._bug_instances
            else:
                logger.debug(f"Sampling {sample_size} from {len(self._bug_instances)} cached instances")
                if random_sample:
                    import random
                    if random_seed is not None:
                        random.seed(random_seed)
                    return random.sample(self._bug_instances, sample_size)
                else:
                    return self._bug_instances[:sample_size]

        logger.info("Processing bug instances...")
        
        if hasattr(self.data, 'to_dict'):
            bug_instances = [dict(zip(self.data.features.keys(), instance)) 
                                  for instance in zip(*self.data.to_dict().values())]
        else:
            bug_instances = self.data

        if sample_size is not None and sample_size < len(bug_instances):
            logger.info(f"Sampling {sample_size} instances from {len(bug_instances)} total instances")
            if random_sample:
                import random
                if random_seed is not None:
                    random.seed(random_seed)
                bug_instances = random.sample(bug_instances, sample_size)
            else:
                bug_instances = bug_instances[:sample_size]

        logger.info(f"Processing {len(bug_instances)} bug instances...")

        
        for i, bug in enumerate(bug_instances):
            if i % 100 == 0: 
                logger.debug(f"Processed {i}/{len(bug_instances)} bug instances")
                
            temp_id = str(bug.get('instance_id', i))

            patch = bug.get('patch', '') # patch contains the code fix on the bug
            modified_files = []
            for line in patch.split('\n'):
                if line.startswith('diff --git'):
                    match = re.search(r'diff --git a/(.*) b/(.*)', line)
                    if match:
                        modified_files.append(match.group(2))
            # Keep only code-file ground truths; skip instance if none remain
            modified_files = [p for p in modified_files if isinstance(p, str) and p.endswith(self.extensions)]
            if not modified_files:
                continue
            
            bug['ground_truths'] = modified_files
            
            bug['bug_report'] = bug.get('problem_statement', '')
            code_files = get_code_files(bug['repo'], bug['base_commit'], self.extensions, self.token)
            if 'problem_statement' in bug:
                del bug['problem_statement']
            
            bug_instance = BugInstance(
                instance_id=temp_id,
                repo=bug['repo'],
                base_commit=bug['base_commit'],
                patch=bug['patch'],
                hints_text=bug['hints_text'],
                ground_truths=bug['ground_truths'],
                bug_report=bug['bug_report'],
                code_files=code_files,
            )
            self._bug_instances.append(bug_instance)
        
        self.repos = list(set([bug.repo for bug in self._bug_instances]))
        
        logger.info(f"Found {len(self.repos)} unique repositories")
        
        logger.info(f"Successfully processed all {len(self._bug_instances)} bug instances")
        return self._bug_instances

    def get_token_statistics(self, model="gpt-4o", sample_size=1000):
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
    

