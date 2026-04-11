import streamlit as st
import anthropic


class ClaudeClient:
    def __init__(self) -> None:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        self._client = anthropic.Anthropic(api_key=api_key)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "claude-opus-4-5",
        max_tokens: int = 1000,
    ) -> str:
        message = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text
