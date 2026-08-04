"""
Real LLM extraction with disk cache, concurrency, retries and cost accounting.

Supports two providers, auto-detected from environment:
  OPENROUTER_API_KEY -> OpenRouter  (OpenAI-compatible chat/completions)
  ANTHROPIC_API_KEY  -> Anthropic   (native messages API)

Pricing is fetched live from OpenRouter's /models endpoint rather than hardcoded,
so cost accounting stays correct as models and prices change. Anthropic pricing
falls back to a small static table.

Design constraints that matter:
  - ONE call per document covering ALL clause types. CUAD contracts average ~13k
    tokens; 41 separate calls per document would multiply input cost by 41 for no
    accuracy gain.
  - Responses cached to disk keyed by (provider, model, prompt version, clause set,
    document text). Re-runs are free; a prompt change correctly invalidates. Same
    key discipline as the derivation store itself.
  - The prompt fights one specific failure mode: a clause TOPIC being discussed
    without the OBLIGATION being established.
"""

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

# Fallback only. OpenRouter prices are fetched live. USD per million tokens.
STATIC_PRICING = {
    "claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00},
    "claude-sonnet-4-6":         {"in": 3.00, "out": 15.00},
    "claude-opus-4-1":           {"in": 15.00, "out": 75.00},
}

PROMPT_VERSION = "3"

SYSTEM = (
    "You are a contract analyst. You determine whether a contract ESTABLISHES a "
    "given clause type, not whether it mentions the topic."
)

INSTRUCTION = """For each clause type below, decide whether this contract actually ESTABLISHES that provision.

Answer true ONLY if the contract creates the obligation, right, or restriction.

Answer false if the clause type is:
- merely mentioned, discussed, or referenced without being established
- expressly excluded, disclaimed, waived, or negated
- deferred to a separate agreement that is not part of this document
- described only in recitals or negotiation history

A clause counts as established even if it is worded indirectly or without a heading,
so long as this document creates the obligation.

Clause types:
{clause_list}

Contract:
<contract>
{text}
</contract>

Reply with a JSON object only. Keys are exactly the clause type identifiers above.
Values are true or false. No prose, no markdown fences."""


def _hash(*parts) -> str:
    m = hashlib.sha256()
    for p in parts:
        m.update(str(p).encode())
    return m.hexdigest()[:20]


# ---------------------------------------------------------------- pricing

_PRICE_CACHE: Dict[str, Dict[str, float]] = {}


def openrouter_pricing(model: str) -> Optional[Dict[str, float]]:
    """Fetch USD-per-million-token pricing live. Cached per process."""
    if not _PRICE_CACHE:
        try:
            with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=30) as r:
                data = json.loads(r.read())
            for m in data.get("data", []):
                p = m.get("pricing") or {}
                try:
                    _PRICE_CACHE[m["id"]] = {
                        "in": float(p.get("prompt", 0)) * 1e6,
                        "out": float(p.get("completion", 0)) * 1e6,
                    }
                except (TypeError, ValueError):
                    continue
        except Exception:
            return None
    return _PRICE_CACHE.get(model)


def list_models(filter_str: str = "") -> List[dict]:
    """Helper: print candidate models and prices. `python -c` friendly."""
    openrouter_pricing("")
    rows = [
        {"id": k, "in_per_mtok": round(v["in"], 3), "out_per_mtok": round(v["out"], 3)}
        for k, v in _PRICE_CACHE.items()
        if filter_str.lower() in k.lower()
    ]
    return sorted(rows, key=lambda r: r["in_per_mtok"])


def resolve_pricing(model: str, provider: str) -> Dict[str, float]:
    if provider == "openrouter":
        p = openrouter_pricing(model)
        if p:
            return p
    return STATIC_PRICING.get(model, {"in": float("nan"), "out": float("nan")})


# ---------------------------------------------------------------- usage

class Usage:
    def __init__(self):
        self.calls = 0
        self.cache_hits = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.errors = 0
        self.wall_seconds = 0.0

    def cost_usd(self, model: str, provider: str) -> float:
        p = resolve_pricing(model, provider)
        return self.in_tokens / 1e6 * p["in"] + self.out_tokens / 1e6 * p["out"]

    def summary(self, model: str, provider: str) -> dict:
        c = self.cost_usd(model, provider)
        return {
            "provider": provider,
            "model": model,
            "api_calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.in_tokens,
            "output_tokens": self.out_tokens,
            "errors": self.errors,
            "cost_usd": round(c, 4),
            "cost_per_doc_usd": round(c / max(self.calls, 1), 5),
            "wall_seconds": round(self.wall_seconds, 1),
        }


# ---------------------------------------------------------------- extractor

class LLMClauseExtractor:
    """Class M extraction operator backed by a real model."""

    def __init__(
        self,
        clause_types: List[str],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        cache_dir: str = "cache/llm",
        max_doc_chars: int = 400_000,
        max_retries: int = 4,
    ):
        self.clause_types = clause_types
        self.or_key = os.environ.get("OPENROUTER_API_KEY")
        self.an_key = os.environ.get("ANTHROPIC_API_KEY")

        self.provider = provider or ("openrouter" if self.or_key else "anthropic")
        if self.provider == "openrouter" and not self.or_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        if self.provider == "anthropic" and not self.an_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        default = ("anthropic/claude-sonnet-4.5" if self.provider == "openrouter"
                   else "claude-haiku-4-5-20251001")
        self.model = model or default
        self.name = f"llm:{self.model.split('/')[-1][:24]}"
        self.version = f"{PROMPT_VERSION}:{self.provider}:{self.model}"
        self.cache_dir = cache_dir
        self.max_doc_chars = max_doc_chars
        self.max_retries = max_retries
        self.usage = Usage()
        self.failed_docs: set = set()
        self._reasoning_detected = False
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "failures"), exist_ok=True)

    # -------- prompt / cache --------

    def _build(self, text: str) -> str:
        listing = "\n".join(f"- {c}" for c in self.clause_types)
        return INSTRUCTION.format(clause_list=listing, text=text[: self.max_doc_chars])

    def _cache_path(self, text: str) -> str:
        key = _hash(self.provider, self.model, PROMPT_VERSION,
                    sorted(self.clause_types), text)
        return os.path.join(self.cache_dir, f"{key}.json")

    # -------- transport --------

    def _request(self, prompt: str) -> urllib.request.Request:
        if self.provider == "openrouter":
            body = {
                "model": self.model,
                "max_tokens": 2048,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            }
            if not self._reasoning_detected:
                body["response_format"] = {"type": "json_object"}
            headers = {
                "content-type": "application/json",
                "authorization": f"Bearer {self.or_key}",
                "HTTP-Referer": "https://github.com/shaikh-amer/absence-query-benchmark",
                "X-Title": "absence-query-benchmark",
            }
            url = OPENROUTER_URL
        else:
            body = {
                "model": self.model,
                "max_tokens": 2048,
                "system": SYSTEM,
                "messages": [{"role": "user", "content": prompt}],
            }
            headers = {
                "content-type": "application/json",
                "x-api-key": self.an_key,
                "anthropic-version": "2023-06-01",
            }
            url = ANTHROPIC_URL
        return urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)

    def _call(self, prompt: str) -> dict:
        delay = 2.0
        last = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(self._request(prompt), timeout=240) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (408, 429, 500, 502, 503, 520, 524, 529) \
                        and attempt < self.max_retries - 1:
                    time.sleep(delay); delay *= 2; continue
                detail = ""
                try:
                    detail = e.read().decode()[:300]
                except Exception:
                    pass
                raise RuntimeError(f"HTTP {e.code} from {self.provider}: {detail}") from e
            except Exception as e:
                last = e
                if attempt < self.max_retries - 1:
                    time.sleep(delay); delay *= 2; continue
                raise
        raise RuntimeError(f"retries exhausted: {last}")

    def _unpack(self, resp: dict):
        """Returns (text, input_tokens, output_tokens) across both providers."""
        if self.provider == "openrouter":
            if "error" in resp and not resp.get("choices"):
                raise RuntimeError(f"openrouter error: {resp['error']}")
            text = resp["choices"][0]["message"].get("content") or ""
            u = resp.get("usage") or {}
            return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        text = "".join(b.get("text", "") for b in resp.get("content", []))
        u = resp.get("usage") or {}
        return text, u.get("input_tokens", 0), u.get("output_tokens", 0)

    _REASONING_RE = re.compile(
        r"<(?:think|thinking|reasoning)\b[^>]*>.*?</(?:think|thinking|reasoning)>",
        re.DOTALL,
    )

    @staticmethod
    def _parse(raw: str, expect_keys: Optional[List[str]] = None) -> dict:
        s = raw.strip()
        s = LLMClauseExtractor._REASONING_RE.sub("", s).strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[1].rsplit("```", 1)[0]

        # Find all balanced {...} candidates and try the last plausible one.
        candidates = []
        depth = 0
        start = -1
        for i, ch in enumerate(s):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(s[start:i + 1])
                if depth < 0:
                    depth = 0
        if not candidates:
            raise ValueError("no JSON object in response")

        for frag in reversed(candidates):
            try:
                parsed = json.loads(frag)
            except json.JSONDecodeError:
                continue
            if expect_keys and not any(k in parsed for k in expect_keys):
                continue
            return parsed

        raise ValueError("no parseable JSON candidate with expected keys")

    # -------- single document --------

    def __call__(self, text: str) -> Dict[str, bool]:
        cp = self._cache_path(text)
        if os.path.exists(cp):
            self.usage.cache_hits += 1
            with open(cp) as f:
                cached = json.load(f)
            return {c: bool(cached.get(c, False)) for c in self.clause_types}

        t0 = time.time()
        try:
            resp = self._call(self._build(text))
        except Exception:
            self.usage.errors += 1
            raise
        self.usage.wall_seconds += time.time() - t0
        self.usage.calls += 1

        raw, in_tok, out_tok = self._unpack(resp)
        self.usage.in_tokens += in_tok
        self.usage.out_tokens += out_tok

        if LLMClauseExtractor._REASONING_RE.search(raw) and not self._reasoning_detected:
            self._reasoning_detected = True

        try:
            parsed = self._parse(raw, expect_keys=self.clause_types)
        except Exception:
            self.usage.errors += 1
            fail_dir = os.path.join(self.cache_dir, "failures")
            fail_path = os.path.join(fail_dir, f"{_hash(self.provider, self.model, PROMPT_VERSION, text)}.txt")
            with open(fail_path, "w") as f:
                f.write(raw)
            raise RuntimeError(
                f"Failed to parse LLM response for document; raw saved to {fail_path}"
            )

        result = {c: bool(parsed.get(c, False)) for c in self.clause_types}
        with open(cp, "w") as f:
            json.dump(result, f)
        return result

    # -------- batch prewarm --------

    def prewarm(self, docs, workers: int = 6, progress: bool = True) -> dict:
        """Fill the cache concurrently. Run once; experiments then cost nothing."""
        todo = [(d.doc_id, d.text) for d in docs.values()
                if not os.path.exists(self._cache_path(d.text))]
        if progress:
            print(f"prewarm: {len(todo)} uncached of {len(docs)} docs "
                  f"| {self.provider} | {self.model}")
        if not todo:
            return self.usage.summary(self.model, self.provider)

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(self.__call__, t): (d, t) for d, t in todo}
            for fut in as_completed(futs):
                done += 1
                doc_id, _ = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    self.failed_docs.add(doc_id)
                    if progress:
                        print(f"  ! {doc_id[:40]}: {type(e).__name__}: {e}")
                if progress and done % 10 == 0:
                    print(f"  {done}/{len(todo)}  "
                          f"${self.usage.cost_usd(self.model, self.provider):.3f}")
        return self.usage.summary(self.model, self.provider)


def estimate_cost(docs, model: str, provider: str = "openrouter",
                  chars_per_token: float = 3.8, out_tokens_per_doc: int = 350) -> dict:
    """Pre-flight estimate. Spends nothing."""
    total_chars = sum(len(d.text) for d in docs.values())
    in_tok = total_chars / chars_per_token
    out_tok = out_tokens_per_doc * len(docs)
    p = resolve_pricing(model, provider)
    cost = in_tok / 1e6 * p["in"] + out_tok / 1e6 * p["out"]
    return {
        "provider": provider,
        "model": model,
        "docs": len(docs),
        "price_in_per_mtok": round(p["in"], 3),
        "price_out_per_mtok": round(p["out"], 3),
        "est_input_tokens": int(in_tok),
        "est_output_tokens": int(out_tok),
        "est_cost_usd": round(cost, 2),
        "est_cost_per_doc_usd": round(cost / max(len(docs), 1), 5),
    }
