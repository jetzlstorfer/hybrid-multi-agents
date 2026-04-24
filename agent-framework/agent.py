# Copyright (c) Microsoft. All rights reserved.

import asyncio
import warnings

from agent_framework._feature_stage import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)


from typing import Annotated
from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential
from pydantic import Field

from transcriber import transcribe, unload

"""
Hello Agent — Simplest possible agent

This sample creates a minimal agent using FoundryChatClient via an
Azure AI Foundry project endpoint, and runs it in both non-streaming and streaming modes.

There are XML tags in all of the get started samples, those are used to display the same code in the docs repo.
"""


@tool
def transcribe_audio(
    file_path: Annotated[
        str, Field(description="The file path to the audio file to transcibe.")
    ],
) -> str:
    """Get the transcription of an audio file."""
    return transcribe(file_path)


# <create_agent>
# Module-level so DevUI's directory discovery (`devui . --port 8080`) picks it up.
client = FoundryChatClient(
    project_endpoint="https://slm-to-llm-resource.services.ai.azure.com",
    model="gpt-5.4-mini-1",
    credential=AzureCliCredential(),
)

agent = Agent(
    client=client,
    name="HelloAgent",
    instructions="You are a friendly assistant. Keep your answers brief.",
    tools=[transcribe_audio],
)
# </create_agent>


async def main() -> None:
    try:
        # <run_agent>
        # Non-streaming: get the complete response at once
        result = await agent.run("Transcribe this audio file: Recording.mp3")
        print(f"Agent: {result}")
        # </run_agent>

        # <run_agent_streaming>
        # Streaming: receive tokens as they are generated
        print("Agent (streaming): ", end="", flush=True)
        async for chunk in agent.run("Tell me a one-sentence fun fact.", stream=True):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()
        # </run_agent_streaming>
    finally:
        unload()


if __name__ == "__main__":
    asyncio.run(main())
