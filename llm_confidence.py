"""
llm_confidence.py — query OpenRouter LLM for trade confidence score.

Returns a float in [0.0, 1.0].
Errors return the neutral value 0.5 so trading is never hard-blocked by LLM.
"""
from __future__ import annotations
import asyncio
import json
import logging
import re

import aiohttp

import config

logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_SYSTEM_PROMPT = (
    "You are a quantitative trading risk analyst. "
    "Given a trade setup, output a single JSON object with one key: "
    '"confidence" (float 0.0–1.0). '
    "1.0 = extremely high confidence setup; 0.0 = no edge. "
    "Reply with ONLY the JSON object, no explanation."
)


def _build_prompt(symbol: str, direction: str, strategy: str, context: dict) -> str:
    ctx_str = json.dumps(context, indent=2)
    return (
        f"Evaluate this US equities intraday trade setup:\n"
        f"Symbol: {symbol}\n"
        f"Direction: {direction}\n"
        f"Strategy: {strategy}\n"
        f"Market context:\n{ctx_str}\n\n"
        f'Respond with {{"confidence": <float>}}'
    )


async def get_llm_confidence(
    symbol: str,
    direction: str,
    strategy: str,
    context: dict,
    timeout: float = 8.0,
) -> float:
    """Return LLM confidence in [0.0, 1.0]. Falls back to 0.5 on any error."""
    if not config.OPENROUTER_API_KEY:
        logger.debug("[LLM] No OPENROUTER_API_KEY — returning neutral 0.5")
        return 0.5

    payload = {
        "model": config.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_prompt(symbol, direction, strategy, context)},
        ],
        "max_tokens":   32,
        "temperature":  0.0,
    }
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/ib_algo_trader",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _OPENROUTER_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("[LLM] HTTP %d: %s", resp.status, body[:200])
                    return 0.5
                data = await resp.json()
                raw  = data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("[LLM] request failed: %s", exc)
        return 0.5

    # parse {"confidence": 0.72} — be lenient with whitespace/formatting
    try:
        match = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
        if match:
            val = float(match.group(1))
            val = max(0.0, min(1.0, val))
            logger.debug("[LLM] %s %s %s → confidence %.2f", symbol, direction, strategy, val)
            return val
    except (ValueError, AttributeError):
        pass

    logger.warning("[LLM] could not parse response: %r", raw)
    return 0.5
