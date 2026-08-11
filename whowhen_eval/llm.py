"""Thin LiteLLM wrapper — the only place this package talks to a provider.

Model routing is entirely LiteLLM's. Pass whatever model id LiteLLM
understands and set the matching environment variable:

===========================================  ==========================
model id                                     env var
===========================================  ==========================
``gpt-5.4``, ``gpt-4.1``, ``o3``             ``OPENAI_API_KEY``
``claude-sonnet-4-6``                        ``ANTHROPIC_API_KEY``
``gemini/gemini-3-flash-preview``            ``GEMINI_API_KEY``
``xai/grok-4``                               ``XAI_API_KEY``
``deepseek/deepseek-chat``                   ``DEEPSEEK_API_KEY``
``together_ai/...``, ``openrouter/...``      provider's key
===========================================  ==========================

We deliberately do **not** manage credentials in code: if a call fails
because a key is missing or wrong, LiteLLM's own error message is what
propagates, since it names the provider and the variable it looked for.

``drop_params`` is enabled globally so a parameter one provider doesn't
support (``temperature`` on reasoning models, ``reasoning_effort`` on
non-reasoning ones) is silently dropped rather than turned into a hard
failure mid-sweep.

Multimodal traces are already assembled as OpenAI content parts
(``{"type": "image_url", "image_url": {"url": "data:image/..."}}``) by the
renderers; LiteLLM accepts that shape for every multimodal provider, so
messages pass straight through.
"""
from __future__ import annotations

from typing import Any, Optional

import litellm

# Provider-unsupported kwargs get dropped instead of raising. Set at import
# time so it applies to every call, including retries inside litellm.
litellm.drop_params = True


def generate(
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    reasoning_effort: Optional[str] = None,
    timeout: int = 600,
    num_retries: int = 5,
) -> tuple[str, dict[str, Optional[int]]]:
    """Run one completion and return ``(text, usage)``.

    ``reasoning_effort`` is forwarded only when the caller sets it, so
    non-reasoning models never see the parameter at all.

    Retries are LiteLLM's built-in ones: ``num_retries`` is forwarded as
    ``max_retries``, which LiteLLM hands to the provider SDK's own retry
    layer (exponential backoff on rate limits, timeouts and transient
    5xx). We deliberately do *not* spell it ``num_retries`` at the call
    site: that spelling additionally arms an outer retry wrapper which
    imports ``tenacity`` lazily, and LiteLLM does not declare tenacity as
    a dependency — so on the first transient error it would raise
    "tenacity import failed" instead of retrying. ``max_retries`` is
    LiteLLM's own primary name for this knob and needs no extra package.

    Exceptions propagate to the caller — :mod:`whowhen_eval.run` catches
    them per-trace so one bad row can't kill a sweep.
    """
    kwargs: dict[str, Any] = {}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=num_retries,
        drop_params=True,
        **kwargs,
    )
    return text_of(response), usage_of(response)


def text_of(response: Any) -> str:
    """Extract the assistant text from a LiteLLM ``ModelResponse``.

    Returns ``""`` rather than raising when a provider hands back an empty
    choice list or a ``None`` message — that happens on content filtering
    and on reasoning models that exhaust ``max_tokens`` before emitting a
    visible answer. An empty prediction is a recordable result; a crash is
    not.
    """
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    try:
        choice = choices[0]
    except (IndexError, TypeError):
        return ""
    message = getattr(choice, "message", None)
    if message is None:
        return ""
    return getattr(message, "content", None) or ""


def usage_of(response: Any) -> dict[str, Optional[int]]:
    """Normalise ``response.usage`` to input/output/total token counts.

    Every field degrades to ``None`` when the provider omits usage
    reporting, so downstream cost analysis can tell "zero tokens" apart
    from "not reported".
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    in_tok = getattr(usage, "prompt_tokens", None)
    if in_tok is None:
        in_tok = getattr(usage, "input_tokens", None)
    out_tok = getattr(usage, "completion_tokens", None)
    if out_tok is None:
        out_tok = getattr(usage, "output_tokens", None)

    total: Optional[int] = None
    if in_tok is not None or out_tok is not None:
        total = (in_tok or 0) + (out_tok or 0)

    return {
        "input_tokens": int(in_tok) if in_tok is not None else None,
        "output_tokens": int(out_tok) if out_tok is not None else None,
        "total_tokens": total,
    }
