import os
import time
import json
import hashlib
import datetime
import re
import math
import ollama as _ollama_lib

# Lazy imports for openai.
_OpenAI = None
openai = None
OLLAMA_SCORE_TOP_LOGPROBS = 5


def _ensure_openai():
    global _OpenAI, openai
    if openai is None:
        import openai as _openai_mod
        from openai import OpenAI as _OpenAI_cls
        openai = _openai_mod
        _OpenAI = _OpenAI_cls


def _normalize_choice_token(token):
    if token is None:
        return ""
    return str(token).strip().strip(".,:;!?()[]{}<>\"'").lower()


def _choice_aliases(choice_spec):
    aliases = {}
    for label, meaning in choice_spec.items():
        label_aliases = {
            _normalize_choice_token(label),
            _normalize_choice_token(f" {label}"),
        }
        if meaning is not None:
            label_aliases.add(_normalize_choice_token(meaning))
            label_aliases.add(_normalize_choice_token(f" {meaning}"))
        aliases[str(label)] = {alias for alias in label_aliases if alias}
    return aliases


def extract_choice_label(text, choice_spec):
    alias_map = {}
    for label, aliases in _choice_aliases(choice_spec).items():
        for alias in aliases:
            alias_map[alias] = label

    raw_text = str(text or "").strip()
    if not raw_text:
        return None

    # Fast path: check the very first character (constrained single-token output)
    first_char = _normalize_choice_token(raw_text[0])
    if first_char in alias_map:
        return alias_map[first_char]

    # Fallback: try the full text and word-level tokens
    candidates = [raw_text]
    candidates.extend(re.findall(r"[A-Za-z]+", raw_text))

    for candidate in candidates:
        normalized = _normalize_choice_token(candidate)
        if normalized in alias_map:
            return alias_map[normalized]
    return None

def score_choice_from_logprobs(logprobs, choice_spec, response_text=""):
    choices = [str(choice) for choice in choice_spec.keys()]
    option_probs = {str(label): 0.0 for label in choices}
    predicted_label = extract_choice_label(response_text, choice_spec)
    confidence_label = predicted_label
    confidence = None
    entropy = None
    status = "missing_logprobs"
    raw_status = status
    fallback_applied = False
    fallback_labels = []
    fallback_logprob = None

    if logprobs:
        first_entry = logprobs[0]
        raw_logprobs = {}
        candidates = list(getattr(first_entry, "top_logprobs", None) or [])
        candidates.append(first_entry)
        for entry in candidates:
            token = _normalize_choice_token(getattr(entry, "token", None))
            logprob = getattr(entry, "logprob", None)
            if not token or logprob is None:
                continue
            if token not in raw_logprobs or logprob > raw_logprobs[token]:
                raw_logprobs[token] = float(logprob)

        aliases = _choice_aliases(choice_spec)
        label_logprobs = {}
        missing_labels = []
        for label, label_aliases in aliases.items():
            matches = [raw_logprobs[alias] for alias in label_aliases if alias in raw_logprobs]
            if matches:
                label_logprobs[label] = max(matches)
            else:
                label_logprobs[label] = float("-inf")
                missing_labels.append(label)

        # Only consider logprobs of tokens that matched a choice label
        # (not all tokens in top_logprobs) for the complement calculation.
        matched_logprobs = [
            lp for lp in label_logprobs.values()
            if math.isfinite(lp)
        ]
        if missing_labels and matched_logprobs:
            fallback_applied = True
            fallback_labels = list(missing_labels)
            # Compute complement probability: if we know P(matched labels),
            # the remaining mass goes to missing labels.
            known_probs = [math.exp(lp) for lp in matched_logprobs]
            total_known = sum(known_probs)
            remaining = max(1.0 - total_known, 1e-12)
            per_missing = remaining / len(missing_labels)
            fallback_logprob = math.log(per_missing)
            for label in fallback_labels:
                label_logprobs[label] = fallback_logprob
            missing_labels = []

        finite_logprobs = {
            label: logprob
            for label, logprob in label_logprobs.items()
            if math.isfinite(logprob)
        }
        if finite_logprobs:
            raw_status = "available" if not fallback_applied else "partial"
            status = "available"
            max_logprob = max(finite_logprobs.values())
            exp_scores = {
                label: math.exp(logprob - max_logprob)
                for label, logprob in finite_logprobs.items()
            }
            denom = sum(exp_scores.values())
            if denom > 0:
                option_probs = {
                    label: (exp_scores.get(label, 0.0) / denom)
                    for label in option_probs
                }
                confidence_label = max(option_probs, key=option_probs.get)
                confidence = float(option_probs[confidence_label])
                if predicted_label is None:
                    predicted_label = confidence_label

                num_options = max(len(option_probs), 1)
                raw_entropy = -sum(
                    prob * math.log(prob)
                    for prob in option_probs.values()
                    if prob > 0
                )
                entropy = (
                    raw_entropy / math.log(num_options)
                    if num_options > 1
                    else 0.0
                )
        else:
            status = "missing_option_logprobs"
            raw_status = status

    uncertainty = None if confidence is None else 1.0 - confidence
    return {
        "score_status": status,
        "raw_score_status": raw_status,
        "choice_spec": {str(label): meaning for label, meaning in choice_spec.items()},
        "predicted_label": predicted_label,
        "confidence_label": confidence_label,
        "option_probs": option_probs,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "entropy": entropy,
        "fallback_applied": fallback_applied,
        "fallback_labels": fallback_labels,
        "fallback_logprob": fallback_logprob,
        "response_text": response_text,
    }


def score_choice_from_raw_logprobs(raw_logprobs_dict, choice_spec, response_text=""):
    """Like score_choice_from_logprobs but takes a plain {token: logprob} dict
    (as stored in the disk cache) instead of OpenAI SDK logprob objects."""

    class _FakeEntry:
        def __init__(self, token, logprob):
            self.token = token
            self.logprob = logprob
            self.top_logprobs = None

    if not raw_logprobs_dict:
        return score_choice_from_logprobs(None, choice_spec, response_text)

    entries = [_FakeEntry(tok, lp) for tok, lp in raw_logprobs_dict.items()]
    entries[0].top_logprobs = entries[1:] if len(entries) > 1 else []
    return score_choice_from_logprobs(entries, choice_spec, response_text)


def score_choice_from_samples(samples, choice_spec, response_text=""):
    """Monte-Carlo estimate of the option distribution for backends without logprobs.

    ``samples`` is a list of raw response strings (one per repeated query). Each option's
    probability is its frequency of being chosen across the samples; confidence is the
    probability of the modal option. This estimates the same quantity that
    score_choice_from_logprobs reads exactly from logprobs.
    """
    labels = [str(label) for label in choice_spec.keys()]
    counts = {label: 0 for label in labels}
    parsed = 0
    for text in (samples or []):
        label = extract_choice_label(text, choice_spec)
        if label is not None and str(label) in counts:
            counts[str(label)] += 1
            parsed += 1

    total = len(samples) if samples else 0
    base = {
        "raw_score_status": f"sampled_{total}",
        "choice_spec": {str(label): meaning for label, meaning in choice_spec.items()},
        "fallback_applied": False,
        "fallback_labels": [],
        "fallback_logprob": None,
        "response_text": response_text,
        "n_samples": total,
        "n_parsed": parsed,
    }
    if total == 0 or parsed == 0:
        predicted = extract_choice_label(response_text, choice_spec)
        base.update({
            "score_status": "missing_samples",
            "predicted_label": predicted,
            "confidence_label": predicted,
            "option_probs": {label: 0.0 for label in labels},
            "confidence": None,
            "uncertainty": None,
            "entropy": None,
        })
        return base

    option_probs = {label: counts[label] / total for label in labels}
    predicted = max(option_probs, key=option_probs.get)
    confidence = float(option_probs[predicted])
    num_options = max(len(option_probs), 1)
    raw_entropy = -sum(p * math.log(p) for p in option_probs.values() if p > 0)
    entropy = raw_entropy / math.log(num_options) if num_options > 1 else 0.0
    base.update({
        "score_status": "sampled",
        "predicted_label": predicted,
        "confidence_label": predicted,
        "option_probs": option_probs,
        "confidence": confidence,
        "uncertainty": 1.0 - confidence,
        "entropy": entropy,
    })
    return base


class LocalLLMPrompter:
    """
    LLM prompter using the native Ollama Python client.

    The OpenAI-compatible /v1 endpoint silently drops the ``think`` field, so
    thinking tokens consume the entire token budget and content comes back empty.
    The native client sends ``think`` as a first-class field on /api/chat,
    which is the correct approach per https://docs.ollama.com/capabilities/thinking

    Setup (one-time):
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull qwen3.5:9b
        ollama serve               # starts the daemon on port 11434

    Parameters
    ----------
    model_name : str
        Ollama model tag.  Recommended for RTX 3090 (24 GB VRAM):
          "qwen3.5:27b"  - 17 GB, 256K context
          "qwen3:30b"    - 19 GB, 256K context, MoE (3B active → fast)
          "qwen3.5:9b"   - 6.6 GB, fastest throughput  ← default
    base_url : str
        Ollama server base URL (default: http://localhost:11434).
    """

    # Maps sampling_params keys → Ollama options keys
    _OPTION_MAP = {
        "max_tokens":        "num_predict",
        "temperature":       "temperature",
        "top_p":             "top_p",
        "stop":              "stop",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty":  "presence_penalty",
    }

    def __init__(self, model_name: str = "deepseek-r1:14b",
                 base_url: str = "http://localhost:11434",
                 think: bool = False) -> None:
        self.model_name = model_name
        self.think = think
        # Strip any /v1 suffix - native client uses the bare host
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self._client = _ollama_lib.Client(host=self.base_url)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_client", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._client = _ollama_lib.Client(host=self.base_url)

    def clone_with(self, **overrides):
        return type(self)(
            model_name=overrides.get("model_name", self.model_name),
            base_url=overrides.get("base_url", self.base_url),
            think=overrides.get("think", self.think),
        )

    def query(self, prompt: dict, sampling_params: dict, save: bool = False, save_dir: str = None, choice_spec= None):
        """
        Query the Ollama model.

        Parameters
        ----------
        prompt          : dict with keys 'system' and 'user'
        sampling_params : dict  (max_tokens, temperature, top_p, stop,...)

        Returns
        -------
        (text: str, score_metadata: dict | None)
        """

        options = {self._OPTION_MAP[k]: v
                   for k, v in sampling_params.items()
                   if k in self._OPTION_MAP}

        # When think mode is active, num_predict must cover both thinking
        # tokens *and* the final answer.  Boost the limit so the model can
        # reason freely and still produce a content response.
        if self.think and "num_predict" in options:
            options["num_predict"] = max(options["num_predict"], 4096)

        messages = []
        if prompt.get("system"):
            messages.append({"role": "system", "content": prompt["system"]})
        messages.append({"role": "user", "content": prompt.get("user", "")})

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self._client.chat(
                    model=self.model_name,
                    messages=messages,
                    think=self.think,
                    logprobs=True,
                    top_logprobs=OLLAMA_SCORE_TOP_LOGPROBS,
                    options=options if options else None,
                    stream=False,
                )
                text = response.message.content
                score_metadata = None
                if choice_spec is not None:
                    score_metadata = score_choice_from_logprobs(
                        getattr(response, "logprobs", None),
                        choice_spec,
                        response_text=text,
                    )
                    if score_metadata["score_status"] == "missing_logprobs":
                        # Logprobs unavailable (e.g. think mode on reasoning models).
                        # Fall back to text-based label extraction with uniform confidence.
                        predicted = extract_choice_label(text, choice_spec)
                        choices = list(choice_spec.keys())
                        n = max(len(choices), 1)
                        score_metadata.update({
                            "score_status": "text_fallback",
                            "raw_score_status": "missing_logprobs",
                            "predicted_label": predicted,
                            "confidence_label": predicted,
                            "confidence": 1.0 / n if predicted is not None else None,
                            "uncertainty": (1.0 - 1.0 / n) if predicted is not None else None,
                            "entropy": 1.0,
                            "option_probs": {str(label): 1.0 / n for label in choices},
                        })

                if save and save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    out_path = os.path.join(save_dir, "response.json")
                    existing = {}
                    if os.path.exists(out_path):
                        with open(out_path) as _f:
                            existing = json.load(_f)
                    existing[self.make_key()] = {
                        "prompt": prompt,
                        "sampling_params": sampling_params,
                        "response": text,
                        "score_metadata": score_metadata,
                    }
                    with open(out_path, "w") as _f:
                        json.dump(existing, _f, indent=4)

                return text, score_metadata

            except Exception as e:
                if isinstance(e, RuntimeError) and "logprobs" in str(e).lower():
                    raise
                print(f"[LocalLLMPrompter] attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(2)

        raise RuntimeError(f"[LocalLLMPrompter] all {max_retries} attempts failed.")

    def make_key(self):
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class BaseOpenAICompatPrompter:
    """Shared logic for any prompter backed by an OpenAI-compatible client.

    Subclasses must set ``self._client``, ``self.model``, and
    ``self._supports_responses_api`` in their ``__init__``.
    """

    OPENAI_TOP_LOGPROBS = 5

    # ── response cache ─────────────────────────────────────────────────────
    def _cache_key(self, prompt, sampling_params, response_model, choice_spec):
        blob = json.dumps({
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "prompt": prompt,
            "sampling_params": sampling_params,
            "response_model": response_model.__name__ if response_model else None,
            "choice_spec": choice_spec,
        }, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def _serialize_logprobs(logprobs):
        """Extract {token: logprob} dict from OpenAI SDK logprob objects."""
        if not logprobs:
            return None
        raw = {}
        first_entry = logprobs[0]
        candidates = list(getattr(first_entry, "top_logprobs", None) or [])
        candidates.append(first_entry)
        for entry in candidates:
            token = getattr(entry, "token", None)
            lp = getattr(entry, "logprob", None)
            if token is not None and lp is not None:
                raw[str(token)] = float(lp)
        return raw

    def _cache_get(self, key, choice_spec=None):
        entry = self._mem_cache.get(key)
        if entry is None and self.cache_dir:
            path = os.path.join(self.cache_dir, f"{key}.json")
            if os.path.exists(path):
                with open(path) as f:
                    entry = json.load(f)
                self._mem_cache[key] = entry
        if entry is None:
            return None
        score_metadata = None
        if choice_spec is not None and entry.get("raw_logprobs"):
            score_metadata = score_choice_from_raw_logprobs(
                entry["raw_logprobs"], choice_spec, response_text=entry["text"],
            )
        return {"text": entry["text"], "score_metadata": score_metadata}

    def _cache_put(self, key, text, raw_logprobs):
        entry = {"text": text, "raw_logprobs": raw_logprobs}
        self._mem_cache[key] = entry
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            path = os.path.join(self.cache_dir, f"{key}.json")
            with open(path, "w") as f:
                json.dump(entry, f, indent=2, default=str)

    # ── compat retries ─────────────────────────────────────────────────────
    @staticmethod
    def _unsupported_param_from_error(exc):
        message = str(exc)
        unsupported_param = re.search(r"Unsupported parameter: '([^']+)'", message)
        if unsupported_param:
            return unsupported_param.group(1)
        unsupported_value = re.search(r"Unsupported value: '([^']+)'", message)
        if unsupported_value:
            return unsupported_value.group(1)
        # SDK TypeError for truly unknown keyword arguments
        if isinstance(exc, TypeError):
            unknown_kwarg = re.search(r"got an unexpected keyword argument '([^']+)'", message)
            if unknown_kwarg:
                return unknown_kwarg.group(1)
        return None

    def _is_gpt5_family(self):
        return str(self.model).lower().startswith("gpt-5")

    def _prestrip_unsupported_params(self, kwargs, *, structured: bool):
        kwargs = dict(kwargs)
        cached_unsupported = self._unsupported_parse_params if structured else self._unsupported_chat_params
        for param in cached_unsupported:
            kwargs.pop(param, None)
        if self._is_gpt5_family():
            known_unsupported = {"temperature", "top_p"}
            if not structured:
                known_unsupported.update({"stop", "frequency_penalty", "presence_penalty"})
            for param in known_unsupported:
                if param in kwargs:
                    kwargs.pop(param, None)
                    cached_unsupported.add(param)
        return kwargs

    def _request_with_compat_retries(self, request_fn, kwargs, *, structured: bool):
        kwargs = self._prestrip_unsupported_params(kwargs, structured=structured)
        tried_removals = set()
        cached_unsupported = self._unsupported_parse_params if structured else self._unsupported_chat_params
        while True:
            try:
                return request_fn(**kwargs)
            except Exception as exc:
                unsupported_param = self._unsupported_param_from_error(exc)
                if unsupported_param and unsupported_param in kwargs and unsupported_param not in tried_removals:
                    tried_removals.add(unsupported_param)
                    kwargs.pop(unsupported_param, None)
                    cached_unsupported.add(unsupported_param)
                    continue
                raise

    # ── reasoning kwargs ───────────────────────────────────────────────────
    def _build_reasoning_kwargs(self):
        kwargs = {}
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
            if self.reasoning_effort == "none":
                kwargs["text"] = {"verbosity": "low"}
        return kwargs

    @staticmethod
    def _extract_logprobs(response):
        output_msg = response.output[0] if response.output else None
        content = output_msg.content[0] if output_msg and output_msg.content else None
        if content and hasattr(content, "logprobs") and content.logprobs:
            return content.logprobs
        return None

    # ── query ──────────────────────────────────────────────────────────────
    def _query_responses_api(self, messages, sampling_params, response_model, choice_spec,
                             save, save_dir, prompt):
        """Call responses.parse (OpenAI structured-output path)."""
        kwargs = {
            "top_logprobs": self.OPENAI_TOP_LOGPROBS,
            "include": ["message.output_text.logprobs"],
        }
        if "max_tokens" in sampling_params:
            kwargs["max_output_tokens"] = max(sampling_params["max_tokens"], 16)
        if "temperature" in sampling_params:
            kwargs["temperature"] = sampling_params["temperature"]
        if "top_p" in sampling_params:
            kwargs["top_p"] = sampling_params["top_p"]
        kwargs.update(self._build_reasoning_kwargs())

        response = self._request_with_compat_retries(
            lambda **call_kwargs: self._client.responses.parse(
                model=self.model,
                input=messages,
                text_format=response_model,
                **call_kwargs,
            ),
            kwargs,
            structured=True,
        )
        parsed = response.output_parsed

        score_metadata = None
        if choice_spec is not None:
            logprobs = self._extract_logprobs(response)
            output_msg = response.output[0] if response.output else None
            content = output_msg.content[0] if output_msg and output_msg.content else None
            response_text = (content.text or "").strip() if content else ""
            score_metadata = score_choice_from_logprobs(logprobs, choice_spec, response_text=response_text)

        if save and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, "response.json")
            existing = {}
            if os.path.exists(out_path):
                with open(out_path) as fh:
                    existing = json.load(fh)
            existing[self.make_key()] = {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "response": parsed.model_dump(),
            }
            with open(out_path, "w") as fh:
                json.dump(existing, fh, indent=4)

        return parsed, score_metadata

    def _query_chat_api(self, messages, sampling_params, choice_spec, cache_key, response_model=None):
        """Call chat.completions.create (universal fallback path).

        When response_model is given, requests JSON-schema structured output
        and parses the response into the Pydantic model before returning.
        """
        chat_args = {
            "model": self.model,
            "messages": messages,
        }
        if response_model is None:
            chat_args["logprobs"] = True
            chat_args["top_logprobs"] = self.OPENAI_TOP_LOGPROBS
        else:
            chat_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                    "strict": False,
                },
            }
        if "max_tokens" in sampling_params:
            # Use max_completion_tokens (the modern standard); fall back to
            # max_tokens automatically via _request_with_compat_retries if
            # the provider rejects it.
            chat_args["max_completion_tokens"] = max(sampling_params["max_tokens"], 16)
        if "temperature" in sampling_params:
            chat_args["temperature"] = sampling_params["temperature"]
        if "top_p" in sampling_params:
            chat_args["top_p"] = sampling_params["top_p"]
        chat_args.update(self._build_reasoning_kwargs())

        response = self._request_with_compat_retries(
            lambda **kw: self._client.chat.completions.create(**kw),
            chat_args,
            structured=False,
        )

        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            msg = getattr(choice, "message", None)
            text = msg.content if (msg and hasattr(msg, "content")) else getattr(choice, "text", "")
            raw_logprobs = getattr(choice, "logprobs", None)
            # OpenAI / Portkey return a ChoiceLogprobs object with a .content
            # list; score_choice_from_logprobs expects a plain indexable list.
            if raw_logprobs is not None and hasattr(raw_logprobs, "content"):
                logprobs = raw_logprobs.content
            else:
                logprobs = raw_logprobs
        else:
            text = ""
            logprobs = None

        score_metadata = None
        if choice_spec is not None:
            score_metadata = score_choice_from_logprobs(logprobs, choice_spec, response_text=text)

        if response_model is not None:
            parsed = response_model.model_validate_json(text)
            return parsed, score_metadata

        if cache_key is not None:
            self._cache_put(cache_key, text, self._serialize_logprobs(logprobs))

        return text, score_metadata

    def query(self, prompt: dict, sampling_params: dict,
              save: bool = False, save_dir: str = None, response_model=None, choice_spec=None):
        # ── Cache lookup (skip for structured output) ─────
        cache_key = None
        if response_model is None:
            cache_key = self._cache_key(prompt, sampling_params, response_model, choice_spec)
            cached = self._cache_get(cache_key, choice_spec=choice_spec)
            if cached is not None:
                return cached["text"], cached.get("score_metadata")

        messages = []
        if prompt.get("system"):
            messages.append({"role": "system", "content": prompt["system"]})
        messages.append({"role": "user", "content": prompt.get("user", "")})

        if self._supports_responses_api and response_model is not None:
            return self._query_responses_api(
                messages, sampling_params, response_model, choice_spec, save, save_dir, prompt
            )

        return self._query_chat_api(messages, sampling_params, choice_spec, cache_key,
                                    response_model=response_model)

    def make_key(self):
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class OpenAILLMPrompter(BaseOpenAICompatPrompter):
    """Prompter using the OpenAI API directly."""

    def __init__(self, api_key: str, model: str = "gpt-5.4", reasoning_effort: str = None,
                 cache_dir: str = None):
        _ensure_openai()
        self.model = model
        self._api_key = api_key
        self.reasoning_effort = reasoning_effort
        self._client = _OpenAI(api_key=api_key)
        self._supports_responses_api = (
            hasattr(self._client, "responses") and hasattr(self._client.responses, "parse")
        )
        self._unsupported_chat_params = set()
        self._unsupported_parse_params = set()
        self.cache_dir = cache_dir
        self._mem_cache = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_client", None)
        state.pop("_mem_cache", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        _ensure_openai()
        self._client = _OpenAI(api_key=self._api_key)
        self._supports_responses_api = (
            hasattr(self._client, "responses") and hasattr(self._client.responses, "parse")
        )
        self._mem_cache = {}


class PortkeyLLMPrompter(BaseOpenAICompatPrompter):
    """Prompter using the native Portkey SDK.

    Parameters
    ----------
    portkey_api_key : str
        Your Portkey account API key (PORTKEY_API_KEY).
    virtual_key : str, optional
        A Portkey virtual key that maps to a specific provider (e.g. OpenAI).
        If omitted, Portkey routes using your account defaults.
    """

    def __init__(self, portkey_api_key: str, model: str = "gpt-5.4",
                 reasoning_effort: str = None, cache_dir: str = None,
                 virtual_key: str = None):
        from portkey_ai import Portkey

        self.model = model
        self._portkey_api_key = portkey_api_key
        self._virtual_key = virtual_key
        self.reasoning_effort = reasoning_effort
        pk_kwargs = {"api_key": portkey_api_key}
        if virtual_key is not None:
            pk_kwargs["virtual_key"] = virtual_key
        self._client = Portkey(**pk_kwargs)
        self._supports_responses_api = False
        self._unsupported_chat_params = set()
        self._unsupported_parse_params = set()
        self.cache_dir = cache_dir
        self._mem_cache = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_client", None)
        state.pop("_mem_cache", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        from portkey_ai import Portkey
        pk_kwargs = {"api_key": self._portkey_api_key}
        if getattr(self, "_virtual_key", None) is not None:
            pk_kwargs["virtual_key"] = self._virtual_key
        self._client = Portkey(**pk_kwargs)
        self._supports_responses_api = False
        self._mem_cache = {}


class AnthropicLLMPrompter:
    """Prompter for the Anthropic Messages API (Claude).

    The Messages API does not return token logprobs, so multiple-choice confidence cannot
    be read directly. Instead, when a ``choice_spec`` is supplied the option distribution
    is estimated by Monte-Carlo sampling: the model is queried ``n_samples`` times at
    ``temperature`` and each option's probability is its frequency of being chosen. This is
    a sampling estimate of the same quantity that the logprob-based backends read exactly,
    so the resulting ``score_metadata`` has the same shape (option_probs / confidence /
    predicted_label / entropy) and is interchangeable downstream.

    Parameters
    ----------
    model : str
        Anthropic model id, e.g. "claude-sonnet-4-6".
    n_samples : int
        Number of samples used to estimate the option distribution (1 -> single greedy
        answer with degenerate confidence 1.0).
    temperature : float
        Sampling temperature (0..1). Use ~1.0 for a faithful estimate of the model's own
        distribution; lower values sharpen it.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6",
                 n_samples: int = 10, temperature: float = 1.0,
                 max_tokens: int = 8, think: bool = False) -> None:
        self.model = model
        self._api_key = api_key
        self.n_samples = max(1, int(n_samples))
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self.think = think
        self._client = self._make_client()

    def _make_client(self):
        import anthropic
        return anthropic.Anthropic(api_key=self._api_key)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_client", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._client = self._make_client()

    def clone_with(self, **overrides):
        return type(self)(
            api_key=overrides.get("api_key", self._api_key),
            model=overrides.get("model", self.model),
            n_samples=overrides.get("n_samples", self.n_samples),
            temperature=overrides.get("temperature", self.temperature),
            max_tokens=overrides.get("max_tokens", self.max_tokens),
            think=overrides.get("think", self.think),
        )

    def _one(self, system: str, user: str, temperature: float) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = max(0.0, min(1.0, float(temperature)))

        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = self._client.messages.create(**kwargs)
                return "".join(
                    getattr(b, "text", "") for b in resp.content
                    if getattr(b, "type", None) == "text"
                ).strip()
            except Exception as e:
                print(f"[AnthropicLLMPrompter] attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"[AnthropicLLMPrompter] all {max_retries} attempts failed.")

    def query(self, prompt: dict, sampling_params: dict, save: bool = False,
              save_dir: str = None, choice_spec=None, response_model=None):
        system = prompt.get("system", "")
        user = prompt.get("user", "")

        if choice_spec is None:
            text = self._one(system, user, temperature=sampling_params.get("temperature", 0.0))
            return text, None

        # Sample the answer n_samples times to estimate the option distribution.
        temp = self.temperature if self.n_samples > 1 else 0.0
        samples = [self._one(system, user, temperature=temp) for _ in range(self.n_samples)]
        score_metadata = score_choice_from_samples(
            samples, choice_spec, response_text=samples[0] if samples else "",
        )

        if save and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, "response.json")
            existing = {}
            if os.path.exists(out_path):
                with open(out_path) as fh:
                    existing = json.load(fh)
            existing[self.make_key()] = {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "samples": samples,
                "score_metadata": score_metadata,
            }
            with open(out_path, "w") as fh:
                json.dump(existing, fh, indent=4)

        return (samples[0] if samples else ""), score_metadata

    def make_key(self):
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
