"""
Job Title Search Service — AI-Powered (OpenAI)
-----------------------------------------------
Uses OpenAI API to dynamically return related job titles.
Handles typos, partial input, and semantic understanding automatically.

Requires: OPENAI_API_KEY environment variable
"""

import json
from openai import OpenAI

_client = OpenAI()

_SYSTEM_PROMPT = """\
You are a job title classification expert. When given a search query (which may contain \
typos or be incomplete), you identify the intended job title and return a JSON object containing a list of \
related job titles.

Rules:
- Always interpret the query charitably — correct typos and partial input.
- Include the closest match to the query FIRST in the list (exact match or closest interpretation).
- ALWAYS include 4 to 14 additional related job titles after the first one, even if there is an exact match.
- "Related" means: same seniority level in a nearby domain, a specialization, a generalization, or a role with significantly overlapping responsibilities.
- Return ONLY a valid JSON object with a single key "results" which contains an array of objects. No markdown, no explanation.
- Each object in the "results" array must have exactly one key: "job_title" (string).
- Return between 5 and 15 results total. Never return fewer than 5.

Example output for "backend developer":
{
  "results": [
    {"job_title": "Backend Developer"},
    {"job_title": "Full Stack Developer"},
    {"job_title": "Software Engineer"},
    {"job_title": "API Developer"},
    {"job_title": "Cloud Engineer"},
    {"job_title": "DevOps Engineer"}
  ]
}
"""


def search_job_titles(query: str, limit: int = 10) -> list[dict]:
    """
    Return a list of job titles related to the query using OpenAI API.

    Args:
        query: User search string — typos and partial input are fine.
        limit: Max number of results to return.

    Returns:
        List of dicts with keys: job_title
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
                "content": (
                    f'Search query: "{query}"\n\n'
                    f'Return the closest matching job title FIRST, then ALWAYS add '
                    f'at least 4 related job titles. Ensure the JSON object has a "results" array with at least 5 items total.'
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()

    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                results = value
                break
        else:
            results = [parsed] if parsed else []
    else:
        results = parsed if isinstance(parsed, list) else [parsed]

    return results[:limit]