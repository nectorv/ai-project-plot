from __future__ import annotations

from langchain_openai import ChatOpenAI

from .session_store import Turn

_PROMPT = """\
You are a query resolver for a data visualization assistant.

Given a conversation history and a new user query, rewrite the query as a \
complete, self-contained data request that requires no prior context to understand.

Rules:
- If the query already makes complete sense on its own, return it UNCHANGED.
- If it references previous results (e.g. "now add X", "do the same for Y", \
"instead show Z", "what about Italy"), rewrite it to include all necessary \
context from the history.
- Return ONLY the rewritten query — no explanation, no punctuation changes \
unless needed.

Conversation history:
{history}

New query: {query}

Rewritten query:\
"""


async def resolve_query(query: str, history: list[Turn]) -> str:
    """
    Rewrite `query` into a self-contained request using `history` as context.
    Returns `query` unchanged if history is empty or the query is already
    self-contained.
    """
    if not history:
        return query

    # Use only the last 3 exchanges (6 turns) to keep the prompt small
    recent = history[-6:]
    history_text = "\n".join(
        f"- {'User' if t.role == 'user' else 'Assistant'}: {t.content}"
        for t in recent
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = _PROMPT.format(history=history_text, query=query)
    result = await llm.ainvoke(prompt)
    resolved = result.content.strip()
    return resolved if resolved else query
