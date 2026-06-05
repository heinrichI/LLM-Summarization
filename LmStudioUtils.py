import requests
from typing import Optional


def get_context_length_from_lm_studio(base_url: str, model_key: str) -> Optional[int]:
    """
    Gets the actual loaded context_length for a model from LM Studio's API.

    Args:
        base_url: LM Studio server URL, e.g. "http://localhost:1234"
        model_key: The model key/ID, e.g. "t-pro-it-2.0"

    Returns:
        The context_length if found, or None if not found/error.
    """
    try:
        url = base_url.rstrip("/") + "/api/v1/models"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        models = data.get("models", data)
        if isinstance(models, dict):
            models = [models]
        for model in models:
            if isinstance(model, dict) and model.get("key") == model_key:
                instances = model.get("loaded_instances", [])
                if instances:
                    return instances[0].get("config", {}).get("context_length")
                else:
                    # Model loaded but no instances? fallback to max_context_length
                    return model.get("max_context_length")

        print(f"Model '{model_key}' not found in LM Studio response.")
        return None

    except requests.RequestException as e:
        print(f"Error connecting to LM Studio API at {base_url}: {e}")
        return None
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error parsing LM Studio API response: {e}")
        return None


if __name__ == "__main__":
    # Quick test
    result = get_context_length_from_lm_studio("http://localhost:1234", "t-pro-it-2.0")
    print(f"context_length: {result}")