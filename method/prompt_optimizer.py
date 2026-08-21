"""BRaIn/SAMMO-style mutation step for the chunk-relevance-judgment prompt (see
method/prompt.py's DEFAULT_CHUNK_RELEVANCE_TEMPLATE). Hand-rolled rather than pulling in
the actual SAMMO library -- MN5 has no internet to install a new package, and this loop
needs to run identically local/MN5, so the mutation itself is just one more structured
LLM call using the same OllamaLocalizer.invoke_structured pattern already used everywhere
else in this pipeline.

Only the natural-language instruction text ever gets rewritten -- the output schema
(ChunkRelevanceJudgment/ChunkRelevanceFeedbackResponse) and the required template
placeholders (method.prompt.CHUNK_RELEVANCE_TEMPLATE_PLACEHOLDERS) are held fixed, so a
mutation can never require changes anywhere downstream of prompt generation.
"""

from method.models import PromptVariantResponse
from method.prompt import CHUNK_RELEVANCE_TEMPLATE_PLACEHOLDERS


def template_has_all_placeholders(template: str) -> bool:
    """A mutated template that dropped/renamed a required placeholder would raise
    KeyError at .format() time deep inside prompt generation -- check up front instead,
    so a malformed variant can be scored as a failure (0 F1) by the search loop rather
    than crashing it."""
    return all(f"{{{p}}}" in template for p in CHUNK_RELEVANCE_TEMPLATE_PLACEHOLDERS)


def generate_prompt_variant(current_template: str, failure_examples: list[dict], meta_localizer, prompt_gen) -> tuple[str, str]:
    """One mutation step: ask `meta_localizer` to rewrite `current_template` given a
    handful of its real mistakes on the dev set. Returns (new_template, rationale).

    Returns (current_template, "mutation failed, reused parent") unchanged if the LLM
    call fails or the response drops a required placeholder -- callers should still
    check the returned template against `template_has_all_placeholders` before scoring
    it, since this is a last-resort fallback, not a guarantee.
    """
    meta_prompt = prompt_gen.generate_prompt_mutation_meta_prompt(current_template, failure_examples)
    try:
        response = meta_localizer.invoke_structured(meta_prompt, PromptVariantResponse)
    except Exception:
        return current_template, "mutation call failed, reused parent template"

    new_template = getattr(response, "prompt_template", None)
    rationale = getattr(response, "rationale", "")
    if not new_template or not template_has_all_placeholders(new_template):
        return current_template, "mutation dropped a required placeholder, reused parent template"

    return new_template, rationale
