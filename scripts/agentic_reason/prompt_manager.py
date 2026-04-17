from typing import List, Dict, Any


def get_instruction_and_prompt(
    dataset_name: str,
    question: Any,
    model_path: str = None,
    max_search_limit: int = 5,
    question_title: str = None,
    tool_use_type: str = "prompt",
    **kwargs,
) -> tuple:
    if isinstance(question, list) and question and isinstance(question[0], list):
        first_turn_messages = question[0]
        question_str = "".join(
            [f"{message['role'].capitalize()}: {message['content']}\n" for message in first_turn_messages]
        )
        question = question_str.strip()
    elif isinstance(question, list) and question and isinstance(question[0], str):
        question = question[0]

    return "", question


def prepare_active_sequence(
    example_data: Dict[str, Any],
    dataset_name: str,
    model_path: str,
    tokenizer,
    tool_use_type: str = "prompt",
    max_search_limit: int = 5,
    use_fewshot_example: bool = False,
) -> list:
    question = example_data["question"]
    instruction, user_prompt = get_instruction_and_prompt(
        dataset_name,
        question,
        model_path,
        max_search_limit,
        tool_use_type=tool_use_type,
    )

    prompt = []
    if "System" in example_data:
        prompt.append({"role": "system", "content": example_data["System"]})

    prompt.append({"role": "user", "content": instruction + user_prompt})

    if tokenizer is not None:
        prompt = tokenizer.apply_chat_template(
            prompt,
            chat_template="{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}",
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt
