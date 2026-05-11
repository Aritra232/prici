"""
Industry Search Service — AI-Powered (OpenAI)
-----------------------------------------------
Uses OpenAI API to dynamically return related industries.
Handles typos, partial input, and semantic understanding automatically.

Requires: OPENAI_API_KEY environment variable
"""

import json
from openai import OpenAI

_client = OpenAI()  # reads OPENAI_API_KEY from environment automatically

_SYSTEM_PROMPT = """\
You are an industry classification expert. When given a search query (which may contain \
typos or be incomplete), you identify the intended industry and return a JSON object containing a list of \
related industries.

Rules:
- Always interpret the query charitably — correct typos and partial input.
- Include the closest match to the query FIRST in the list.
- ALWAYS include 4 to 14 additional related industries after the first one, even if there is an exact match.
- Return ONLY a valid JSON object with a single key "results" which contains an array of objects. No markdown, no explanation.
- Each object in the "results" array must have exactly two keys: "industry" (string) and "reason" (short string explaining the relation).
- Return between 5 and 15 results total. Never return fewer than 5.

Example output format:
{
  "results": [
    {"industry": "Healthcare", "reason": "Direct match"},
    {"industry": "Pharmaceuticals", "reason": "Drug development within healthcare"},
    {"industry": "Medical Devices", "reason": "Equipment used in healthcare"},
    {"industry": "Biotechnology", "reason": "Biological research for healthcare"},
    {"industry": "Health Informatics", "reason": "IT services for healthcare"}
  ]
}
"""


def search_industries(query: str, limit: int = 10) -> list[dict]:
    """
    Return a list of industries related to the query using OpenAI API.

    Args:
        query: User search string — typos and partial input are fine.
        limit: Max number of results to return.

    Returns:
        List of dicts with keys: industry, reason
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
                "content": f'Search query: "{query}"\n\nReturn the exact match FIRST if applicable, then return 4-10 additional related industries. Ensure the JSON object has a "results" array with at least 5 items total.',
            },
        ],
        response_format={"type": "json_object"},  # enforces JSON output
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()

    parsed = json.loads(raw)
    if isinstance(parsed, dict):
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