import re
from typing import List, Dict, Optional
from vllm import SamplingParams
from omegaconf import DictConfig
import logging


def extract_answer(output, mode="gen"):
    extracted_text = ""
    if mode == "codegen":
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()
    elif mode == "infogen":
        pattern_info = "**Final Information**"
        pattern_step = "**Modified Reasoning Steps**"
        if pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].replace("\n", "").strip("```").strip()
        elif pattern_step in output:
            extracted_text = output.split(pattern_step)[-1].strip("```").strip()
        else:
            extracted_text = "No helpful information found."
    else:
        pattern = r"\\boxed\{(.*)\}"
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]
            if mode in ["choose", "qa"]:
                inner_pattern = r"\\text\{(.*)\}"
                inner_matches = re.findall(inner_pattern, extracted_text)
                if inner_matches:
                    extracted_text = inner_matches[-1]
                extracted_text = extracted_text.strip("()")
    return extracted_text


def generate_webpage_to_reasonchain_batch(
    original_questions: List[str],
    prev_reasonings: List[str],
    search_queries: List[str],
    documents: List[str],
    batch_output_records: List[Dict],
    llm,
    tokenizer,
    max_tokens: int = 32768,
    coherent: bool = False,
) -> List[str]:
    if llm is None:
        extracted_infos = documents
        for i, doc in enumerate(documents):
            batch_output_records.append({"prompt": None, "raw_output": None, "extracted_info": doc})
        return extracted_infos

    # Search-extractor path not used in datasci-only builds; keep minimal fallback.
    extracted_infos = [doc for doc in documents]
    return extracted_infos


def run_generation(
    session,
    llm,
    tokenizer,
    llm_config: DictConfig,
    tools: Optional[List[dict]] = None,
    stop_tokens: List[str] = None,
    add_tool_hint: Optional[bool] = False,
    logger: Optional[logging.Logger] = None,
) -> List:
    repetition_penalty = llm_config.get("repetition_penalty") or 1.1
    max_tokens = llm_config.get("max_tokens") or 8192

    temperature = llm_config.get("temperature") or 0.7
    top_p = llm_config.get("top_p") or 0.8
    top_k = llm_config.get("top_k") or 20
    max_think_tokens = llm_config.get("max_think_tokens") or None

    hint_message = None
    if add_tool_hint and tools:
        tool_names_this_turn = list(tools.keys())
        if tool_names_this_turn:
            hint_message = {
                "role": "user",
                "content": f"Helpful reminder: The tools available for this step are: {tool_names_this_turn}.",
            }
            session.add_to_prompt(hint_message)
            if logger:
                logger.info(f"Injected tool availability hint: {hint_message['content']}")

    prompts = [session.prompt]

    stop_tokens = []
    if tokenizer is not None:
        stop_tokens.append(tokenizer.eos_token)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stop=stop_tokens,
        include_stop_str_in_output=True,
    )

    if max_think_tokens and max_think_tokens > 0:
        extra_kwargs = {"max_think_tokens": max_think_tokens}
    else:
        extra_kwargs = {}

    output_list, raw_outputs = llm.generate(prompts, tools=tools, sampling_params=sampling_params, **extra_kwargs)

    if hint_message:
        if session.prompt and session.prompt[-1] == hint_message:
            session.pop_from_prompt()

    return output_list, raw_outputs
