"""
src/generation/generator.py
============================
LLM generation module for the banking RAG pipeline.

Features:
- Multi-provider support (OpenAI GPT-4o, Anthropic Claude)
- Streaming responses
- Retry logic with exponential backoff
- Token usage tracking
- Response validation (hallucination check)
- Citation extraction and verification
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)


# ─── Generation Result ────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Full result from LLM generation."""
    answer: str
    sources_cited: List[str]
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    finish_reason: str
    is_grounded: bool = True           # Did the answer stay within context?
    contains_uncertainty: bool = False  # Did the model express uncertainty?
    metadata: Dict = field(default_factory=dict)

    @property
    def cost_estimate_usd(self) -> float:
        """Estimate API cost in USD (GPT-4o pricing as of 2024)."""
        # GPT-4o: $2.50/1M input tokens, $10.00/1M output tokens
        input_cost = (self.prompt_tokens / 1_000_000) * 2.50
        output_cost = (self.completion_tokens / 1_000_000) * 10.00
        return round(input_cost + output_cost, 6)


# ─── Citation Extractor ───────────────────────────────────────────────────────

def extract_citations(text: str) -> List[str]:
    """
    Extract citations from generated text.
    Looks for patterns like [Source: filename.pdf] or [Source: policy_manual]
    """
    patterns = [
        r'\[Source:\s*([^\]]+)\]',
        r'\(Source:\s*([^\)]+)\)',
        r'According to ([^,\.]+\.(pdf|docx|txt))',
    ]
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            citation = match[0] if isinstance(match, tuple) else match
            citations.append(citation.strip())
    return list(set(citations))


def check_groundedness(answer: str, context: str) -> Tuple[bool, float]:
    """
    Simple heuristic check: is the answer grounded in context?
    
    Returns (is_grounded, confidence_score)
    
    In production, replace with NLI model (e.g., deberta-v3-base-tasksource-nli)
    """
    # Check for uncertainty phrases (good sign - model knows its limits)
    uncertainty_phrases = [
        "not in the documentation",
        "not available",
        "not mentioned",
        "contact your loan officer",
        "please consult",
        "i don't have",
        "cannot confirm",
    ]

    answer_lower = answer.lower()
    has_uncertainty = any(phrase in answer_lower for phrase in uncertainty_phrases)

    # Check if key answer phrases appear in context (simple but effective)
    answer_sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 20]
    grounded_count = 0

    for sentence in answer_sentences[:5]:  # Check first 5 sentences
        # Extract key noun phrases (simplified - use spaCy in production)
        words = set(sentence.lower().split())
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "have", "has", "had", "do", "does", "did", "will", "would",
                     "could", "should", "may", "might", "this", "that", "these",
                     "those", "for", "and", "or", "but", "in", "on", "at", "to"}
        key_words = words - stop_words

        if len(key_words) < 3:
            continue

        # Check if at least 40% of key words appear in context
        context_lower = context.lower()
        matches = sum(1 for w in key_words if w in context_lower)
        if len(key_words) > 0 and matches / len(key_words) >= 0.4:
            grounded_count += 1

    total_checked = min(5, len(answer_sentences))
    score = grounded_count / max(total_checked, 1)
    is_grounded = score >= 0.3 or has_uncertainty  # Either grounded OR expressing uncertainty

    return is_grounded, score


# ─── OpenAI Generator ────────────────────────────────────────────────────────

class OpenAIGenerator:
    """
    GPT-4o powered generator with streaming support.
    Optimized for factual, low-temperature banking responses.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_retries: int = 3,
    ):
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai required: pip install openai")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    async def generate(
        self,
        messages: List[dict],
        system_prompt: str,
        context: str = "",
    ) -> GenerationResult:
        """Generate a response with retry logic."""
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system_prompt}] + messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    # Structured outputs for consistent format
                    response_format={"type": "text"},
                )

                answer = response.choices[0].message.content
                latency = (time.time() - start_time) * 1000

                citations = extract_citations(answer)
                is_grounded, ground_score = check_groundedness(answer, context)
                has_uncertainty = any(
                    phrase in answer.lower()
                    for phrase in ["not in the documentation", "contact your loan officer", "please consult"]
                )

                result = GenerationResult(
                    answer=answer,
                    sources_cited=citations,
                    model=self.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    latency_ms=round(latency, 1),
                    finish_reason=response.choices[0].finish_reason,
                    is_grounded=is_grounded,
                    contains_uncertainty=has_uncertainty,
                    metadata={"ground_score": round(ground_score, 3)},
                )

                logger.info(
                    "generation_complete",
                    model=self.model,
                    tokens=response.usage.total_tokens,
                    latency_ms=round(latency, 1),
                    is_grounded=is_grounded,
                    citations=len(citations),
                    cost_usd=result.cost_estimate_usd,
                )

                return result

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 1.0 * (2 ** attempt)
                    logger.warning("generation_retry", attempt=attempt + 1, error=str(e), wait=wait)
                    await asyncio.sleep(wait)
                else:
                    logger.error("generation_failed", error=str(e))
                    raise

    async def stream(
        self,
        messages: List[dict],
        system_prompt: str,
    ) -> AsyncIterator[str]:
        """Stream response tokens for real-time UI updates."""
        async with self.client.chat.completions.stream(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ) as stream:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content


# ─── Anthropic Generator ──────────────────────────────────────────────────────

class AnthropicGenerator:
    """
    Claude-powered generator (Anthropic).
    Alternative to OpenAI for organizations preferring Claude.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_retries: int = 3,
    ):
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic required: pip install anthropic")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    async def generate(
        self,
        messages: List[dict],
        system_prompt: str,
        context: str = "",
    ) -> GenerationResult:
        """Generate using Claude API."""
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt,
                    messages=messages,
                    temperature=self.temperature,
                )

                answer = response.content[0].text
                latency = (time.time() - start_time) * 1000
                citations = extract_citations(answer)
                is_grounded, ground_score = check_groundedness(answer, context)

                return GenerationResult(
                    answer=answer,
                    sources_cited=citations,
                    model=self.model,
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                    latency_ms=round(latency, 1),
                    finish_reason=str(response.stop_reason),
                    is_grounded=is_grounded,
                    metadata={"ground_score": round(ground_score, 3)},
                )

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1.0 * (2 ** attempt))
                else:
                    raise


# ─── Generator Factory ────────────────────────────────────────────────────────

class GeneratorFactory:
    @staticmethod
    def create(provider: str = "openai", **kwargs):
        if provider == "openai":
            import os
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            return OpenAIGenerator(api_key=api_key, **{k: v for k, v in kwargs.items() if k != "api_key"})
        elif provider == "anthropic":
            import os
            api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            return AnthropicGenerator(api_key=api_key, **{k: v for k, v in kwargs.items() if k != "api_key"})
        else:
            raise ValueError(f"Unknown provider: {provider}")
