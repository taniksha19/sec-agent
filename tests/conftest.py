import os

# Tests never send traces, even when .env has Langfuse keys loaded.
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
