from typing import Optional
from openai import OpenAI
from config import config

SYSTEM_PROMPT = """You are a game asset prompt engineer. Your job is to rewrite a user's brief description into a detailed, professional prompt for AI image generation.

Rules:
1. Expand the description with specific visual details: pose, perspective, art style, color palette, composition.
2. Use standard game-art terminology appropriate to the asset type.
3. Keep output to 2-3 sentences, under 300 characters.
4. Do NOT add phrases like "create an image of..." — output the description directly.
5. Preserve all key nouns and concepts from the original description.
6. Output ONLY the rewritten description, nothing else."""

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
        )
    return _client


def rewrite_prompt(raw_description: str, asset_type: str) -> str:
    """Rewrite a user's raw description into a detailed game-asset prompt.

    Returns the original description unchanged if the API call fails
    or no API key is configured.
    """
    if not config.deepseek_api_key:
        return raw_description

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Asset type: {asset_type}\n"
                        f"User description: {raw_description}\n\n"
                        f"Rewrite this into a detailed game-asset prompt."
                    ),
                },
            ],
            max_tokens=200,
            temperature=0.7,
        )
        rewritten = response.choices[0].message.content
        return rewritten.strip() if rewritten else raw_description
    except Exception:
        return raw_description
