"""
Hobby Search Service — AI-Powered (OpenAI)
--------------------------------------------
Uses OpenAI API to dynamically return related hobbies.
Handles typos, partial input, and semantic understanding automatically.

Requires: OPENAI_API_KEY environment variable
"""

import json
from openai import OpenAI

_client = OpenAI()  # reads OPENAI_API_KEY from environment automatically

_SYSTEM_PROMPT = """\
You are a hobby classification expert. When given a search query (which may contain \
typos or be incomplete), you identify the intended hobby and return a JSON list of \
related hobbies.

Rules:
- Always interpret the query charitably — correct typos and partial input.
- Include the closest match to the query FIRST in the list.
- Include all meaningfully related hobbies (same family, similar activity, etc.).
- Return ONLY a valid JSON array of objects. No markdown, no explanation, no extra text.
- Each object must have exactly two keys: "hobby" (string) and "reason" (short string explaining the relation).
- Return between 5 and 15 results.

Example output format:
[
  {"hobby": "Cycling", "reason": "Direct match"},
  {"hobby": "Mountain Biking", "reason": "Off-road variant of cycling"},
  {"hobby": "Road Cycling", "reason": "Paved-road variant of cycling"}
]
"""


def search_hobbies(query: str, limit: int = 10) -> list[dict]:
    """
    Return a list of hobbies related to the query using OpenAI API.

    Args:
        query: User search string — typos and partial input are fine.
        limit: Max number of results to return.

    Returns:
        List of dicts with keys: hobby, reason
    """
    query = query.strip()
    if not query:
        return []

    response = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f'Search query: "{query}"\n\nReturn the exact match FIRST if applicable, then return 4-5 additional related hobbies as a JSON array with at least 5 items total.',
            },
        ],
        response_format={"type": "json_object"},  # enforces JSON output
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()

    # OpenAI json_object mode wraps in an object — unwrap if needed
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        # model may return {"hobbies": [...]} or {"results": [...]}
        # find the first list value in the dict
        for value in parsed.values():
            if isinstance(value, list):
                results = value
                break
        else:
            # if no list found, wrap the whole dict as a single result
            results = [parsed] if parsed else []
    else:
        results = parsed if isinstance(parsed, list) else [parsed]

    return results[:limit]