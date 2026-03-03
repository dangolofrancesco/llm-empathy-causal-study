"""
Big Five Personality Scoring via LLM-as-a-Judge.

For each user in the input CSV, sends their conversation to an LLM
and asks it to score the five traits (0.0–1.0) as a professional psychiatrist.

Input:  data/user_aggregate_conversations.csv
Output: data/user_big_five_scores.jsonl  (one JSON object per line)
        data/user_aggregate_conversations_scored.csv  (original CSV + trait columns filled in)

Supported providers (set via --provider):
  openai   – requires OPENAI_API_KEY   (default model: gpt-4o)
  mistral  – requires MISTRAL_API_KEY  (default model: mistral-small-latest)
  groq     – requires GROQ_API_KEY     (default model: llama-3.3-70b-versatile)

Usage:
    python3 scripts/score_big_five.py --provider mistral
    python3 scripts/score_big_five.py --provider groq
    python3 scripts/score_big_five.py --provider openai --model gpt-4o
    python3 scripts/score_big_five.py --provider mistral --max_chars 6000 --delay 0.3
    python3 scripts/score_big_five.py --provider mistral --resume --limit 50
"""

import argparse
import json
import os
import time
import re
from pathlib import Path

import pandas as pd
import os
os.environ["MISTRAL_API_KEY"] = "lplY8dzzqH1hPHfXrdj3ACp1H3wPazvf"
# ── Client factory ────────────────────────────────────────────────────────────

PROVIDER_DEFAULTS = {
    "openai":   "gpt-4o",
    "mistral":  "mistral-small-latest",
    "groq":     "llama-3.3-70b-versatile",
}

def build_client(provider: str):
    """Return a client object for the given provider."""
    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run:  pip install openai")
        return OpenAI()   # reads OPENAI_API_KEY

    elif provider == "mistral":
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError("Run:  pip install mistralai")
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError("Set the MISTRAL_API_KEY environment variable.")
        return Mistral(api_key=api_key)

    elif provider == "groq":
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Run:  pip install groq")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("Set the GROQ_API_KEY environment variable.")
        return Groq(api_key=api_key)

    else:
        raise ValueError(f"Unknown provider: {provider!r}. Choose from: openai, mistral, groq")


def call_llm(client, provider: str, model: str, messages: list) -> str:
    """Unified chat-completion call that works for OpenAI, Mistral, and Groq."""
    if provider == "mistral":
        response = client.chat.complete(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=150,
        )
    else:
        # openai and groq share the same SDK interface
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=150,
        )
    return response.choices[0].message.content.strip()


# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert clinical psychologist and psychiatrist specialising in personality assessment.
You will be given a collection of chat messages written by a single user.
Your task is to infer that person's Big Five personality trait scores solely from their written language, vocabulary, topics, tone, and interaction style.

For each of the five traits output a floating-point score between 0.0 (extremely low) and 1.0 (extremely high).
Return ONLY a valid JSON object with exactly these keys, nothing else:
{
  "trait_openness": <float 0.0-1.0>,
  "trait_consciousness": <float 0.0-1.0>,
  "trait_extraversion": <float 0.0-1.0>,
  "trait_agreableness": <float 0.0-1.0>,
  "trait_neuroticism": <float 0.0-1.0>
}"""

USER_PROMPT_TEMPLATE = """Below are all the messages written by this user across their conversations.
Analyse the text carefully and return the Big Five scores as instructed.

--- USER MESSAGES ---
{conversation}
--- END ---"""

TRAITS = [
    "trait_openness",
    "trait_consciousness",
    "trait_extraversion",
    "trait_agreableness",
    "trait_neuroticism",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_json(text: str) -> dict | None:
    """Extract a JSON object from LLM output, even if wrapped in markdown fences."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return None
    return None


def score_user(client, provider: str, conversation: str, model: str, max_chars: int,
               max_retries: int = 10, base_wait: float = 5.0) -> dict | None:
    """Call the LLM and return a dict with the five trait scores, or None on failure.

    On 429 rate-limit errors, retries with exponential back-off up to max_retries times.
    """
    truncated = conversation[:max_chars]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(conversation=truncated)},
    ]
    for attempt in range(1, max_retries + 1):
        try:
            raw = call_llm(client, provider, model, messages)
            return extract_json(raw)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = (
                "429" in err_str
                or "rate_limited" in err_str.lower()
                or "rate limit" in err_str.lower()
                or "too many requests" in err_str.lower()
            )
            if is_rate_limit and attempt < max_retries:
                wait = base_wait * (2 ** (attempt - 1))   # 5s, 10s, 20s, 40s …
                print(f"\n    [RATE LIMIT] attempt {attempt}/{max_retries} – waiting {wait:.0f}s …",
                      end=" ", flush=True)
                time.sleep(wait)
                continue
            print(f"\n    [API ERROR] {e}")
            return None
    return None


def validate_scores(scores: dict) -> dict:
    """Clamp all trait values to [0, 1] and ensure all keys are present."""
    result = {}
    for trait in TRAITS:
        val = scores.get(trait)
        if val is None:
            result[trait] = None
        else:
            try:
                result[trait] = round(max(0.0, min(1.0, float(val))), 4)
            except (TypeError, ValueError):
                result[trait] = None
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score Big Five traits using an LLM-as-a-judge.")
    parser.add_argument("--provider",  default="mistral", choices=["openai", "mistral", "groq"],
                        help="LLM provider to use (default: mistral)")
    parser.add_argument("--input",     default="data/user_aggregate_conversations.csv",
                        help="Path to input CSV (default: data/user_aggregate_conversations.csv)")
    parser.add_argument("--model",     default=None,
                        help="Model name override. Defaults: openai=gpt-4o, mistral=mistral-small-latest, groq=llama-3.3-70b-versatile")
    parser.add_argument("--max_chars", type=int, default=8000,
                        help="Max characters of conversation to send per user (default: 8000)")
    parser.add_argument("--delay",     type=float, default=0.3,
                        help="Seconds to wait between API calls (default: 0.3)")
    parser.add_argument("--resume",    action="store_true",
                        help="Skip users already present in the output JSONL file")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Stop after scoring this many users (default: score all)")
    args = parser.parse_args()

    model = args.model or PROVIDER_DEFAULTS[args.provider]

    root = Path(__file__).parent.parent
    input_path  = root / args.input
    jsonl_path  = root / "data" / "user_big_five_scores.jsonl"
    scored_csv  = root / "data" / "user_aggregate_conversations_scored.csv"

    # ── Load input ─────────────────────────────────────────────────────────────
    print(f"[INFO] Provider : {args.provider}  |  Model: {model}")
    print(f"[INFO] Reading  : {input_path}")
    df = pd.read_csv(input_path)
    print(f"[INFO] {len(df)} users loaded.")

    # ── Resume: load already-scored users ──────────────────────────────────────
    already_scored: dict[str, dict] = {}
    if args.resume and jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    uid = record["user_hashed_ip"]
                    # Only count as done if ALL five traits are present and non-null
                    if all(record.get(t) is not None for t in TRAITS):
                        already_scored[uid] = record
        print(f"[INFO] Resuming: {len(already_scored)} users already fully scored.")

    # ── Build LLM client ───────────────────────────────────────────────────────
    client = build_client(args.provider)

    # ── Score each user ────────────────────────────────────────────────────────
    results: list[dict] = list(already_scored.values())
    scored_count = 0

    if args.limit is not None:
        print(f"[INFO] Limit    : {args.limit} users")

    with open(jsonl_path, "a", encoding="utf-8") as out_f:
        for i, row in df.iterrows():
            if args.limit is not None and scored_count >= args.limit:
                print(f"[INFO] Reached limit of {args.limit} users — stopping.")
                break

            uid  = str(row["user_hashed_ip"])
            conv = str(row.get("conversation", ""))

            if uid in already_scored:
                continue  # already done

            print(f"[{i+1}/{len(df)}] Scoring user {uid[:16]}...", end=" ", flush=True)

            scores = score_user(client, args.provider, conv, model=model, max_chars=args.max_chars)

            if scores is None:
                print("FAILED – no valid JSON returned.")
                record = {"user_hashed_ip": uid, **{t: None for t in TRAITS}}
            else:
                validated = validate_scores(scores)
                record = {"user_hashed_ip": uid, **validated}
                trait_str = "  ".join(f"{t.split('_')[1][:4]}={v}" for t, v in validated.items())
                print(f"OK  {trait_str}")

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            results.append(record)
            scored_count += 1

            time.sleep(args.delay)

    # ── Merge scores back into the original DataFrame ──────────────────────────
    scores_df = pd.DataFrame(results)
    merged = df.drop(columns=[t for t in TRAITS if t in df.columns], errors="ignore")
    merged = merged.merge(scores_df, on="user_hashed_ip", how="left")
    merged.to_csv(scored_csv, index=False)

    print(f"\n[DONE] JSONL scores  → {jsonl_path}")
    print(f"[DONE] Scored CSV    → {scored_csv}")
    print(merged[["user_hashed_ip", *TRAITS]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
