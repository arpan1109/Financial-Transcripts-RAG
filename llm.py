import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

FOLLOWUP_INDICATORS = {
    "it", "its", "that", "this", "these", "those", "them", "they",
    "each", "former", "latter", "same", "above", "previous", "again",
    "too", "also", "one", "ones",
}

QUESTION_STARTERS = {
    "what", "why", "how", "where", "who", "which", "when",
    "can", "does", "do", "is", "are", "will", "should",
}

def _looks_like_followup(question: str) -> bool:
    """Heuristic to determine if a question requires prior context to make sense."""
    words = question.lower().replace("?", "").replace(",", "").split()
    if not words:
        return False
    if any(w in FOLLOWUP_INDICATORS for w in words):
        return True
    if words[0] in QUESTION_STARTERS:
        return False
    return len(words) <= 4

def reformulate_query(current_question: str, history: list) -> str:
    """Prepends the last user question if the current one is a follow-up."""
    if not history or not _looks_like_followup(current_question):
        return current_question

    last_user_turn = next(
        (t["content"] for t in reversed(history) if t["role"] == "user"), None
    )
    if not last_user_turn:
        return current_question

    return f"{last_user_turn} {current_question}"

def generate_streaming_answer(question: str, chunks: list, history: list = None):
    """Generates a grounded, streaming answer with inline [1] citations using a Dual-LLM fallback."""
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        src = chunk.metadata.get("source", "unknown")
        qtr = chunk.metadata.get("quarter", "")
        yr = chunk.metadata.get("year", "")
        context_blocks.append(f"[{i}] (Source: {src} {qtr} {yr})\n{chunk.page_content}")

    context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant documents found."

    history_text = ""
    if history:
        recent = history[-4:] 
        history_text = "\n".join(f"{t['role']}: {t['content']}" for t in recent)

    system_prompt = """You are FinSight AI, a precise financial equity research assistant.
Answer ONLY using the provided context chunks. Answer ONLY the current question being asked.
For every factual claim or number, cite the source using its bracket number, e.g., [1] or [2].
If the context doesn't fully answer the CURRENT question, say so explicitly. Do not invent metrics."""

    user_prompt = f"""Conversation so far:
{history_text}

Context:
{context_text}

Question: {question}

Answer the question using only the context above, with [n] citations."""

    # DUAL-LLM FALLBACK LOGIC RESTORED
    models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    
    for model in models:
        try:
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                stream=True,
            )
        except Exception as e:
            if model == models[-1]:
                raise e # Throw error if the final fallback model fails
            continue

def extract_kpis(context: str) -> dict:
    """Extracts background metrics for the dashboard using Dual-LLM fallback."""
    sys_msg = """Output ONLY valid JSON:
    {"revenue": "$ value", "margin": "% value", "net_income": "$ value", "eps": "$ value"}
    Use 'N/A' if absent."""
    
    models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    
    for model in models:
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": context}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(res.choices[0].message.content)
        except Exception as e:
            if model == models[-1]:
                raise e
            continue

def parse_stream(stream):
    """Generator for Streamlit UI."""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content