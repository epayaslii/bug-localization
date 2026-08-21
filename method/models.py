from pydantic import BaseModel
from typing import List


class OpenAILocalizerResponse(BaseModel):
    candidate_files: list[str]


class DirectorySelectionResponse(BaseModel):
    directories_to_explore: List[str] = []
    selected_files: List[str] = []


class FileRelevanceJudgment(BaseModel):
    file: str
    relevant: bool


class RelevanceFeedbackResponse(BaseModel):
    judgments: List[FileRelevanceJudgment]


class ChunkRelevanceJudgment(BaseModel):
    file: str
    chunk_index: int
    relevant: bool


class ChunkRelevanceFeedbackResponse(BaseModel):
    judgments: List[ChunkRelevanceJudgment]


class PromptVariantResponse(BaseModel):
    prompt_template: str
    rationale: str


class ChunkReference(BaseModel):
    file: str
    chunk_index: int


class SparseChunkRelevanceResponse(BaseModel):
    """Alternative to ChunkRelevanceFeedbackResponse: lists only the chunks judged
    relevant, instead of a relevant/not-relevant verdict for every chunk. Dramatically
    fewer output tokens for the same information on a large candidate pool (most chunks
    in a ~200-chunk pool are irrelevant, so a dense response spends most of its token
    budget restating "false") -- confirmed empirically to matter: the dense format only
    covered 37/211 chunks before running out of max_tokens on one real dev-set instance.
    Any chunk NOT listed here is treated as not relevant by omission."""
    relevant_chunks: List[ChunkReference]


