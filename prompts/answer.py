"""Prompt templates used by answer-generation features."""

from string import Formatter


ANSWER = (
    "You are a support assistant. Answer ONLY from the context.\n"
    "If the answer is not present, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def render(template: str, **values: str) -> str:
    """Fill a template's named placeholders with runtime values."""
    fields = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None
    }
    missing = fields - values.keys()
    if missing:
        raise ValueError(f"missing template values: {sorted(missing)}")
    unexpected = values.keys() - fields
    if unexpected:
        raise ValueError(f"unexpected template values: {sorted(unexpected)}")
    return template.format(**values)


def build_answer_prompt(context: str, question: str) -> str:
    """Render the shared answer prompt for a question and retrieved context."""
    return render(ANSWER, context=context, question=question)