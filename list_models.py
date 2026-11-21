"""Print the model IDs your API key can use, so you can pick one for LLM_MODEL.

    python list_models.py
"""

from qa.llm import DEFAULT_MODEL, LLMClient


def main() -> None:
    client = LLMClient()
    model_ids = sorted(model.id for model in client.client.models.list())
    print(f"Current LLM_MODEL: {client.model}\n")
    print("Models available to your key:")
    for model_id in model_ids:
        marker = "  <- default" if model_id == DEFAULT_MODEL else ""
        print(f"  {model_id}{marker}")
    if client.model not in model_ids:
        print(f"\n'{client.model}' is not in this list. Set LLM_MODEL in .env to one of the IDs above.")


if __name__ == "__main__":
    main()
