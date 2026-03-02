from openai import OpenAI


def create_openai_client(*, api_key: str, base_url: str) -> OpenAI:
    """Build a shared OpenAI-compatible client."""
    return OpenAI(api_key=api_key, base_url=base_url)
