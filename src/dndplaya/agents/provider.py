"""LLM provider abstraction — Anthropic API and Ollama (OpenAI-compatible).

Providers translate between BaseAgent's internal message format and
the wire format of each backend.  BaseAgent delegates all API calls
to whichever provider is configured.
"""
from __future__ import annotations

import json
import logging
import time as _time
from dataclasses import dataclass, field
from typing import Protocol

from ..config import Settings

logger = logging.getLogger(__name__)


def _parse_tool_args(
    args_str: str,
    param_names: list[str],
    properties: dict,
) -> dict:
    """Parse a tool call's argument string into a dict.

    Handles:
      - Named kwargs:  target="goblin", amount=-5
      - Positional args: "goblin", "perception", "medium"
      - JSON-style:    monsters=[{"name": "goblin", "cr": 0.25}]
      - Empty args:    (nothing)
    """
    if not args_str:
        return {}

    # Try JSON object parse first (handles complex args)
    try:
        result = json.loads("{" + args_str + "}")
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Try as a JSON array of positional args
    try:
        values = json.loads("[" + args_str + "]")
        if isinstance(values, list) and param_names:
            return {
                param_names[i]: v
                for i, v in enumerate(values)
                if i < len(param_names)
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # Named kwargs:  key="value", key2=value2
    import re
    kwargs_pattern = r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\[[^\]]*\])|(\{[^}]*\})|([^,\s]+))'
    kwargs_matches = re.findall(kwargs_pattern, args_str)
    if kwargs_matches:
        result = {}
        for m in kwargs_matches:
            key = m[0]
            # Pick the first non-empty group as the value
            val_str = m[1] or m[2] or m[3] or m[4] or m[5]
            # Try to parse as JSON for lists/dicts/numbers
            try:
                result[key] = json.loads(val_str)
            except (json.JSONDecodeError, ValueError):
                result[key] = val_str
        if result:
            return result

    # Single string argument
    stripped = args_str.strip().strip("\"'")
    if param_names and stripped:
        return {param_names[0]: stripped}

    return {}


# ── Normalised response ─────────────────────────────────────────────

@dataclass
class ToolCall:
    """A tool call from the API response."""

    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Provider-agnostic response that BaseAgent consumes."""

    text_parts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_content: list[dict] = field(default_factory=list)
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    thinking: str | None = None


# ── Provider protocol ────────────────────────────────────────────────

class LLMProvider(Protocol):
    """Interface that every backend must satisfy."""

    def call(
        self,
        messages: list[dict],
        system: str | list,
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
        thinking_budget: int = 0,
    ) -> LLMResponse: ...


# ── Anthropic provider ───────────────────────────────────────────────

class AnthropicProvider:
    """Wraps the Anthropic Python SDK (prompt caching, thinking, retry)."""

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0  # seconds

    def __init__(self, settings: Settings):
        import anthropic
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
        )
        self.model = settings.model

    # -- public interface --------------------------------------------------

    def call(
        self,
        messages: list[dict],
        system: str | list,
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
        thinking_budget: int = 0,
    ) -> LLMResponse:
        # Decorate system prompt + messages for caching
        cached_system = self._add_cache_control(system)
        msgs = list(messages)
        self._mark_last_for_caching(msgs)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": cached_system,
            "messages": msgs,
            "timeout": 120.0,
        }
        if tools:
            kwargs["tools"] = tools
        if enable_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            kwargs["max_tokens"] = max(kwargs["max_tokens"], thinking_budget + 2048)

        response = self._call_with_retry(kwargs)
        return self._parse_response(response)

    # -- caching helpers (kept as static for testability) ------------------

    @staticmethod
    def _add_cache_control(system_prompt: str | list) -> list:
        """Wrap system prompt with cache_control for prompt caching."""
        if isinstance(system_prompt, str):
            return [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
        result = [dict(block) for block in system_prompt]
        if result:
            result[-1]["cache_control"] = {"type": "ephemeral"}
        return result

    @staticmethod
    def _mark_last_for_caching(messages: list) -> None:
        """Add cache_control to the last message's content block."""
        if not messages:
            return
        last = messages[-1]
        content = last["content"]
        if isinstance(content, str):
            last["content"] = [{
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }]
        elif isinstance(content, list) and content:
            new_content = list(content)
            last_block = new_content[-1]
            if isinstance(last_block, dict):
                new_content[-1] = {**last_block, "cache_control": {"type": "ephemeral"}}
            last["content"] = new_content

    # -- retry logic -------------------------------------------------------

    def _call_with_retry(self, kwargs: dict):
        anthropic = self._anthropic
        last_error: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return self.client.messages.create(**kwargs)
            except (
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.RateLimitError,
                anthropic.InternalServerError,
            ) as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    _time.sleep(self.RETRY_BASE_DELAY * (2 ** attempt))
        raise last_error  # type: ignore[misc]

    # -- response parsing --------------------------------------------------

    def _parse_response(self, response) -> LLMResponse:
        from anthropic.types import TextBlock, ToolUseBlock, ThinkingBlock

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_content: list[dict] = []
        thinking: str | None = None

        for block in (response.content or []):
            if isinstance(block, ThinkingBlock):
                thinking = block.thinking
                raw_content.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature,
                })
            elif isinstance(block, TextBlock):
                if block.text.strip():
                    text_parts.append(block.text)
                    raw_content.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(ToolCall(
                    id=block.id, name=block.name, arguments=block.input,
                ))
                raw_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        # Usage
        usage = response.usage
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        return LLMResponse(
            text_parts=text_parts,
            tool_calls=tool_calls,
            raw_content=raw_content,
            stop_reason=response.stop_reason,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            thinking=thinking,
        )


# ── Ollama provider (OpenAI-compatible) ──────────────────────────────

class OllamaProvider:
    """Speaks the OpenAI-compatible API that Ollama exposes at /v1."""

    MAX_RETRIES = 3          # retries on connection errors
    MAX_TOOL_RETRIES = 2     # retries on malformed tool calls
    RETRY_BASE_DELAY = 1.0

    def __init__(self, settings: Settings):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for Ollama support. "
                "Install it with: pip install openai"
            )
        self._openai = openai
        self.client = openai.OpenAI(
            base_url=f"{settings.ollama_url}/v1",
            api_key="ollama",  # Ollama ignores this but the SDK requires it
        )
        self.model = settings.ollama_model

    # -- public interface --------------------------------------------------

    def call(
        self,
        messages: list[dict],
        system: str | list,
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
        thinking_budget: int = 0,
    ) -> LLMResponse:
        # Translate everything to OpenAI format
        oai_messages = self._translate_messages(messages, system)
        oai_tools = self._translate_tools(tools) if tools else None

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": oai_messages,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools

        # Call with tool-validation retry loop
        for attempt in range(self.MAX_TOOL_RETRIES + 1):
            response = self._call_with_retry(kwargs)
            llm_resp = self._parse_response(response)

            # Local models often write tool calls as text instead of using
            # the function-calling API.  If we got text but no tool_calls,
            # try to extract them from the response text.
            if not llm_resp.tool_calls and tools and llm_resp.text_parts:
                extracted = self._extract_text_tool_calls(
                    "\n".join(llm_resp.text_parts), tools,
                )
                if extracted:
                    llm_resp.tool_calls = extracted
                    # Rebuild raw_content with extracted tool calls
                    for tc in extracted:
                        llm_resp.raw_content.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                    llm_resp.stop_reason = "tool_use"

            if not llm_resp.tool_calls or not tools:
                return llm_resp

            errors = self._validate_tool_calls(llm_resp.tool_calls, tools)
            if not errors:
                return llm_resp

            if attempt < self.MAX_TOOL_RETRIES:
                logger.warning(
                    "Ollama tool validation failed (attempt %d/%d): %s",
                    attempt + 1, self.MAX_TOOL_RETRIES, errors,
                )
                # Append the bad response + error feedback and retry
                kwargs["messages"] = list(kwargs["messages"])
                kwargs["messages"].append(
                    self._assistant_msg_from_response(response)
                )
                kwargs["messages"].append({
                    "role": "user",
                    "content": (
                        f"Your tool call had errors: {errors}\n"
                        "Please try again with a valid tool call."
                    ),
                })
            else:
                # Exhausted retries — return text-only so session doesn't crash
                logger.error(
                    "Ollama tool validation failed after %d retries: %s — "
                    "returning text-only response",
                    self.MAX_TOOL_RETRIES, errors,
                )
                llm_resp.tool_calls = []
                return llm_resp

        return llm_resp  # unreachable but satisfies type checker

    # -- message translation -----------------------------------------------

    @staticmethod
    def _translate_messages(messages: list[dict], system: str | list) -> list[dict]:
        """Convert BaseAgent internal messages to OpenAI chat format."""
        oai: list[dict] = []

        # System prompt
        if isinstance(system, str):
            sys_text = system
        elif isinstance(system, list):
            sys_text = "\n\n".join(
                block.get("text", "") for block in system
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            sys_text = str(system)
        oai.append({"role": "system", "content": sys_text})

        # Conversation messages
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                oai.extend(OllamaProvider._translate_user_message(content))
            elif role == "assistant":
                oai.append(OllamaProvider._translate_assistant_message(content))

        return oai

    @staticmethod
    def _translate_user_message(content) -> list[dict]:
        """Translate a user message (may contain tool_results or images)."""
        if isinstance(content, str):
            return [{"role": "user", "content": content}]

        if not isinstance(content, list):
            return [{"role": "user", "content": str(content)}]

        # Check if it's all tool_result blocks
        tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
        if tool_results and len(tool_results) == len(content):
            # All tool results → individual tool messages
            return [
                {
                    "role": "tool",
                    "tool_call_id": tr["tool_use_id"],
                    "content": tr.get("content", ""),
                }
                for tr in tool_results
            ]

        # Mixed content (text + images + possibly tool_results)
        oai_parts: list[dict] = []
        tool_msgs: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                oai_parts.append({"type": "text", "text": block.get("text", "")})
            elif btype == "image":
                # Anthropic base64 image → OpenAI image_url
                source = block.get("source", {})
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                oai_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data}"},
                })
            elif btype == "tool_result":
                tool_msgs.append({
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"],
                    "content": block.get("content", ""),
                })

        result: list[dict] = []
        if oai_parts:
            result.append({"role": "user", "content": oai_parts})
        result.extend(tool_msgs)
        return result

    @staticmethod
    def _translate_assistant_message(content) -> dict:
        """Translate an assistant message (may contain tool_use blocks)."""
        if isinstance(content, str):
            return {"role": "assistant", "content": content}

        if not isinstance(content, list):
            return {"role": "assistant", "content": str(content)}

        text_parts: list[str] = []
        tool_calls: list[dict] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })
            # Skip thinking blocks — Ollama doesn't understand them

        msg: dict = {"role": "assistant", "content": "\n".join(text_parts) or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    # -- tool translation --------------------------------------------------

    @staticmethod
    def _translate_tools(tools: list[dict]) -> list[dict]:
        """Anthropic tool format → OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    # -- tool validation ---------------------------------------------------

    @staticmethod
    def _validate_tool_calls(
        tool_calls: list[ToolCall],
        tools: list[dict],
    ) -> list[str]:
        """Validate tool calls against the tool schema. Returns list of error strings."""
        tool_map = {t["name"]: t for t in tools}
        errors: list[str] = []

        for tc in tool_calls:
            if tc.name not in tool_map:
                available = ", ".join(sorted(tool_map.keys()))
                errors.append(f"Unknown tool '{tc.name}'. Available: {available}")
                continue

            schema = tool_map[tc.name].get("input_schema", {})
            required = schema.get("required", [])
            properties = schema.get("properties", {})

            # Check required params
            for param in required:
                if param not in tc.arguments:
                    errors.append(
                        f"Tool '{tc.name}' missing required param '{param}'"
                    )

            # Check enum values
            for param, value in tc.arguments.items():
                if param in properties:
                    prop = properties[param]
                    if "enum" in prop and value not in prop["enum"]:
                        errors.append(
                            f"Tool '{tc.name}' param '{param}' value "
                            f"'{value}' not in enum {prop['enum']}"
                        )

        return errors

    # -- text-based tool call extraction -----------------------------------

    @staticmethod
    def _extract_text_tool_calls(
        text: str,
        tools: list[dict],
    ) -> list[ToolCall]:
        """Extract tool calls that the model wrote as text instead of using the API.

        Matches patterns like:
            narrate("some text here")
            ask_skill_check("Thorin", "perception", "medium")
            roll_initiative(monsters=[{"name": "goblin", "cr": 0.25}])
            request_group_input()
        """
        import re
        import uuid

        tool_map = {t["name"]: t for t in tools}

        # --- Phase 1: Extract JSON-in-text tool calls ---
        # Qwen often writes: {"name": "say", "arguments": {"text": "...", "urgency": 3}}
        # Sometimes wrapped in XML-like tags or prefixed with garbage text.
        extracted: list[ToolCall] = []
        json_pattern = r'\{"name":\s*"(\w+)",\s*"arguments":\s*(\{[^}]*\})\}'
        for match in re.finditer(json_pattern, text):
            name = match.group(1)
            if name not in tool_map:
                continue
            try:
                arguments = json.loads(match.group(2))
            except (json.JSONDecodeError, ValueError):
                arguments = {}
            extracted.append(ToolCall(
                id=f"text_{uuid.uuid4().hex[:8]}",
                name=name,
                arguments=arguments,
            ))

        # If JSON extraction found anything, return those (they're higher quality)
        if extracted:
            return extracted

        # --- Phase 2: Extract function-call-style tool calls ---
        tool_names_pattern = "|".join(re.escape(name) for name in tool_map)
        # Match tool_name(...) — also handles wrapping parens like (tool_name())
        # and bare (tool_name) without inner parens
        pattern = rf'(?:\(?\s*)({tool_names_pattern})\s*\(([^)]*(?:\([^)]*\)[^)]*)*)\)\s*\)?'
        # Also match bare tool names without parens, e.g. "(request_group_input)"
        bare_pattern = rf'\(({tool_names_pattern})\)'

        for match in re.finditer(pattern, text):
            name = match.group(1)
            args_str = match.group(2).strip()

            schema = tool_map[name].get("input_schema", {})
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            param_names = list(properties.keys())

            arguments = _parse_tool_args(args_str, param_names, properties)

            # Fill in defaults for missing required params where we can
            # (e.g., empty string for string params)
            for param in required:
                if param not in arguments and param in properties:
                    ptype = properties[param].get("type", "string")
                    if ptype == "string":
                        arguments[param] = ""
                    elif ptype == "integer":
                        arguments[param] = 0
                    elif ptype == "boolean":
                        arguments[param] = False

            extracted.append(ToolCall(
                id=f"text_{uuid.uuid4().hex[:8]}",
                name=name,
                arguments=arguments,
            ))

        # Also match bare tool names like (request_group_input) without ()
        seen_names = {tc.name for tc in extracted}
        for match in re.finditer(bare_pattern, text):
            name = match.group(1)
            if name in seen_names:
                continue  # already found via the primary pattern
            schema = tool_map[name].get("input_schema", {})
            required = schema.get("required", [])
            properties = schema.get("properties", {})
            arguments: dict = {}
            for param in required:
                if param in properties:
                    ptype = properties[param].get("type", "string")
                    if ptype == "string":
                        arguments[param] = ""
                    elif ptype == "integer":
                        arguments[param] = 0
                    elif ptype == "boolean":
                        arguments[param] = False
            extracted.append(ToolCall(
                id=f"text_{uuid.uuid4().hex[:8]}",
                name=name,
                arguments=arguments,
            ))
            seen_names.add(name)

        return extracted

    # -- retry logic -------------------------------------------------------

    def _call_with_retry(self, kwargs: dict):
        openai = self._openai
        last_error: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return self.client.chat.completions.create(**kwargs)
            except (
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    _time.sleep(self.RETRY_BASE_DELAY * (2 ** attempt))
        raise last_error  # type: ignore[misc]

    # -- response parsing --------------------------------------------------

    def _parse_response(self, response) -> LLMResponse:
        """Parse an OpenAI-format response into LLMResponse."""
        choice = response.choices[0]
        msg = choice.message

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_content: list[dict] = []

        if msg.content:
            text_parts.append(msg.content)
            raw_content.append({"type": "text", "text": msg.content})

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
                raw_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": args,
                })

        # Map OpenAI stop reasons to our format
        stop_reason = "end_turn"
        if choice.finish_reason == "tool_calls":
            stop_reason = "tool_use"

        # Token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return LLMResponse(
            text_parts=text_parts,
            tool_calls=tool_calls,
            raw_content=raw_content,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @staticmethod
    def _assistant_msg_from_response(response) -> dict:
        """Build an OpenAI assistant message dict from a raw response (for retry)."""
        choice = response.choices[0]
        msg = choice.message
        result: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return result


# ── Factory ──────────────────────────────────────────────────────────

def create_provider(settings: Settings) -> LLMProvider:
    """Create the appropriate LLM provider based on settings."""
    provider = getattr(settings, "provider", "anthropic")
    if provider == "ollama":
        return OllamaProvider(settings)  # type: ignore[return-value]
    return AnthropicProvider(settings)  # type: ignore[return-value]
