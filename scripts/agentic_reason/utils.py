import re
from typing import Dict, Optional, Any
import copy
import json

from agentic_reason.config import BEGIN_SEARCH_QUERY, BEGIN_SEARCH_RESULT

_TYPE_ALIAS: Dict[str, str] = {
    # 基础类型
    "float":   "number",
    "double":  "number",
    "number":  "number",
    "int":     "integer",
    "integer": "integer",
    "long":    "integer",
    "short":   "integer",
    "str":     "string",
    "string":  "string",
    "text":    "string",
    "char":    "string",
    "bool":    "boolean",
    "boolean": "boolean",
    "bytes":   "string",
    "byte":    "string",
    "any":     "string",      # BFCL treats "any" as a free-form string

    # 容器类型
    "list":    "array",
    "tuple":   "array",
    "array":   "array",
    "set":     "array",

    "dict":    "object",
    "object":  "object",
    "map":     "object",

    # 复合/特殊类型
    "none":    "null",
    "null":    "null",
    "nonetype": "null",
    "date":    "string",
    "datetime": "string",
    "timestamp": "string",

    # 兼容大小写
    "Float":   "number",
    "Double":  "number",
    "Int":     "integer",
    "Integer": "integer",
    "Long":    "integer",
    "Short":   "integer",
    "Str":     "string",
    "String":  "string",
    "Text":    "string",
    "Char":    "string",
    "Bool":    "boolean",
    "Boolean": "boolean",
    "Bytes":   "string",
    "Byte":    "string",
    "Any":     "string",
    "List":    "array",
    "Tuple":   "array",
    "Array":   "array",
    "Set":     "array",
    "Dict":    "object",
    "Object":  "object",
    "Map":     "object",
    "None":    "null",
    "Null":    "null",
    "NoneType": "null",
    "Date":    "string",
    "Datetime": "string",
    "Timestamp": "string",
}

def _normalise_schema(node: Any) -> None:
    """
    • Replace Gorilla/Python type names with JSON-Schema primitives
    • Inject additionalProperties:false for every object
    • Ensure each 'required' array only names existing properties
    """
    if isinstance(node, dict):
        # -- 1) normalise this node's "type" ---------------------------
        t = node.get("type")
        if isinstance(t, str):
            node["type"] = _TYPE_ALIAS.get(t, t)
        elif isinstance(t, list):
            node["type"] = [_TYPE_ALIAS.get(x, x) for x in t]

        # -- 2) lock down objects -------------------------------------
        if node.get("type") == "object" and "additionalProperties" not in node:
            node["additionalProperties"] = False

        # -- 3) keep 'required' in sync with 'properties' -------------
        props = node.get("properties")
        req = node.get("required")
        if req:
            if not props:
                node.pop("required", None)                # no props → drop
            else:
                filtered = [k for k in req if k in props]
                if filtered:
                    node["required"] = filtered
                else:
                    node.pop("required", None)

        # -- 4) recurse -----------------------------------------------
        for v in node.values():
            if isinstance(v, (dict, list)):
                _normalise_schema(v)

    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                _normalise_schema(item)

def convert_tool_schema(
    raw_schema: Dict[str, Any],
    model: str,
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Convert a Gorilla / BFCL-style function spec into the schema format
    required by OpenAI, Claude, or Qwen.
    """

    # 1) clone & normalise the parameter block ------------------------
    params = copy.deepcopy(raw_schema["parameters"])
    if params.get("type") == "dict":                 # Gorilla quirk
        params["type"] = "object"
    _normalise_schema(params)

    # 2) drop optional args from top-level 'required' -----------------
    orig_required = list(params.get("required", []))
    params["required"] = [
        k for k in orig_required
        if "default" not in params.get("properties", {}).get(k, {})
    ]
    if not params["required"]:
        params.pop("required", None)

    # 3) build the function block ------------------------------------
    fn_block = {
        "name": f"{prefix}{raw_schema['name']}",
        "description": raw_schema["description"],
        "parameters": params,
    }

    # *Only* keep strict if the author set it explicitly
    if raw_schema.get("strict") is True:
        fn_block["strict"] = True

    # 4) wrap for the chosen model family -----------------------------
    lower = model.lower()
    if "claude" in lower:
        return {
            "name": fn_block["name"],
            "description": fn_block["description"],
            "input_schema": params          # Claude ignores 'strict'
        }
    if lower in ("gpt-o4-mini", "gpt-o3"):
        return {"type": "function", **fn_block}
    if "gpt" in lower or "qwen" in lower:
        return {"type": "function", "function": fn_block}

    raise ValueError(f"Unrecognised model family: {model}")

def extract_tool_calls_fc(
    messages: list[dict[str, Any]],
    default_unit: str = "units"
) -> list[dict[str, dict[str, Any]]]:
    """
    Extract tool/function calls from a chat transcript.
    • Handles both Gorilla-style ('content' list with type='tool_use')
      and OpenAI function-calling style ('tool_calls' list).
    """
    calls: list[dict[str, dict[str, Any]]] = []

    for msg in messages:
        if msg.get("role") != "assistant":
            continue

        # ---------- 1) OpenAI function-calling format ----------
        for call in msg.get("tool_calls") or []:
            if call.get("type") != "function":
                continue
            fn_name = call["function"]["name"]
            try:
                params = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                params = {}
            calls.append({fn_name: params})

        # ---------- 2) Gorilla / BFCL legacy format ----------
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "tool_use":
                    fn_name = item["name"]
                    params = dict(item.get("input", {}))
                    calls.append({fn_name: params})

    return calls

def parse_steps(text: str) -> Dict[int, str]:
    """Parse reasoning steps from text into a dictionary."""
    steps = {}
    current_step = None
    current_content = []
    
    for line in text.split('\n'):
        # Try to match a step number at the start of the line
        step_match = re.match(r'^Step\s*(\d+):\s*(.*)$', line.strip())
        
        if step_match:
            # If we were building a previous step, save it
            if current_step is not None:
                steps[current_step] = '\n'.join(current_content).strip()
            
            # Start new step
            current_step = int(step_match.group(1))
            current_content = [f"Step {current_step}: {step_match.group(2)}"]
        elif current_step is not None:
            current_content.append(line)
    
    # Save the last step if exists
    if current_step is not None:
        steps[current_step] = '\n'.join(current_content).strip()
    
    return steps

# def replace_recent_steps(origin_str: str, replace_str: str) -> str:
#     """Replace recent reasoning steps in the original string with new steps."""
#     # Reference to original implementation
#     # Reference lines from run_agentic_reason.py:
    
#     # Parse the original and replacement steps
#     origin_steps = parse_steps(origin_str)
#     replace_steps = parse_steps(replace_str)
    
#     # Apply replacements
#     for step_num, content in replace_steps.items():
#         if "DELETE THIS STEP" in content:
#             # Remove the step if it exists
#             if step_num in origin_steps:
#                 del origin_steps[step_num]
#         else:
#             # Replace or add the step
#             origin_steps[step_num] = content
    
#     # Sort the steps by step number
#     sorted_steps = sorted(origin_steps.items())
    
#     # Reconstruct the reasoning steps as a single string
#     new_reasoning_steps = "\n\n".join([f"{content}" for num, content in sorted_steps])
    
#     return new_reasoning_steps

def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """Extract text between two tags."""
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None

def normalize_url(url: str) -> str:
    """Normalize URL for consistent caching."""
    url = url.strip().lower()
    if url.endswith('/'):
        url = url[:-1]
    return url

def clean_snippet(snippet: str) -> str:
    """Clean snippet text by removing HTML tags and normalizing whitespace."""
    # Remove HTML tags
    clean_text = re.sub('<[^<]+?>', '', snippet)
    # Normalize whitespace
    clean_text = ' '.join(clean_text.split())
    return clean_text

def replace_recent_steps(origin_str, replace_str):
        """
        Replaces specific steps in the original reasoning steps with new steps.
        If a replacement step contains "DELETE THIS STEP", that step is removed.

        Parameters:
        - origin_str (str): The original reasoning steps.
        - replace_str (str): The steps to replace or delete.

        Returns:
        - str: The updated reasoning steps after applying replacements.
        """

        def parse_steps(text):
            """
            Parses the reasoning steps from a given text.

            Parameters:
            - text (str): The text containing reasoning steps.

            Returns:
            - dict: A dictionary mapping step numbers to their content.
            """
            step_pattern = re.compile(r"Step\s+(\d+):\s*")
            steps = {}
            current_step_num = None
            current_content = []

            for line in text.splitlines():
                step_match = step_pattern.match(line)
                if step_match:
                    # If there's an ongoing step, save its content
                    if current_step_num is not None:
                        steps[current_step_num] = "\n".join(current_content).strip()
                    current_step_num = int(step_match.group(1))
                    content = line[step_match.end():].strip()
                    current_content = [content] if content else []
                else:
                    if current_step_num is not None:
                        current_content.append(line)
            
            # Save the last step if any
            if current_step_num is not None:
                steps[current_step_num] = "\n".join(current_content).strip()
            
            return steps

        # Parse the original and replacement steps
        origin_steps = parse_steps(origin_str)
        replace_steps = parse_steps(replace_str)

        # Apply replacements
        for step_num, content in replace_steps.items():
            if "DELETE THIS STEP" in content:
                # Remove the step if it exists
                if step_num in origin_steps:
                    del origin_steps[step_num]
            else:
                # Replace or add the step
                origin_steps[step_num] = content

        # Sort the steps by step number
        sorted_steps = sorted(origin_steps.items())

        # Reconstruct the reasoning steps as a single string
        new_reasoning_steps = "\n\n".join([f"{content}" for num, content in sorted_steps])

        return new_reasoning_steps

def extract_reasoning_context(all_reasoning_steps, mind_map = None) -> str:
        if mind_map:
            truncated_prev_reasoning = mind_map.query("summarize the reasoning process, be short and clear")
            return truncated_prev_reasoning
        else:
            truncated_prev_reasoning = ""
            for i, step in enumerate(all_reasoning_steps):
                truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

            prev_steps = truncated_prev_reasoning.split('\n\n')
            if len(prev_steps) <= 5:
                truncated_prev_reasoning = '\n\n'.join(prev_steps)
            else:
                truncated_prev_reasoning = ''
                for i, step in enumerate(prev_steps):
                    if i == 0 or i >= len(prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                        truncated_prev_reasoning += step + '\n\n'
                    else:
                        if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                            truncated_prev_reasoning += '...\n\n'
            truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')
            return truncated_prev_reasoning
        