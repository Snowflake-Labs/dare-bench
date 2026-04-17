from prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC


def build_action_to_tool_map():
    """BFCL-style action map removed; datasci-eval does not use missed_function."""
    return {}


def setup_dynamic_tool_loading(example_data, is_multi_turn, logger):
    missed_function_config = example_data.get("missed_function", {}) if is_multi_turn else {}
    action_to_tool_map = build_action_to_tool_map()
    if missed_function_config:
        logger.info(f"missed_function config present but action map is empty (datasci-only build).")
    return missed_function_config, action_to_tool_map


def get_current_turn_tools(
    available_tool_schemas,
    available_tool_dispatch,
    missed_function_config,
    action_to_tool_map,
    user_turn_index,
    logger,
):
    """Filter tools per turn when missed_function is set (unused for standard datasci-eval)."""
    current_turn_tools_schemas = available_tool_schemas
    current_turn_tool_dispatch = available_tool_dispatch

    turn_key_for_missed_func = str(user_turn_index + 1)
    if turn_key_for_missed_func in missed_function_config:
        functions_to_miss = missed_function_config[turn_key_for_missed_func]
        tools_to_hide_this_turn = set()

        canonical_tool_names = available_tool_schemas.keys()
        lower_to_canonical_map = {name.lower(): name for name in canonical_tool_names}

        for func_to_miss in functions_to_miss:
            if func_to_miss in action_to_tool_map:
                tool_name_from_map = action_to_tool_map[func_to_miss]
                tool_name_lower = tool_name_from_map.lower()

                if tool_name_lower in lower_to_canonical_map:
                    canonical_name = lower_to_canonical_map[tool_name_lower]
                    tools_to_hide_this_turn.add(canonical_name)
                else:
                    logger.warning(
                        f"Tool name '{tool_name_from_map}' from map for action '{func_to_miss}' not in canonical tool list."
                    )
            else:
                logger.warning(
                    f"Could not map missing function '{func_to_miss}' to any tool in the action map."
                )

        if tools_to_hide_this_turn:
            logger.warning(f"Hiding tools for user turn {user_turn_index + 1}: {list(tools_to_hide_this_turn)}")
            current_turn_tools_schemas = {
                k: v for k, v in available_tool_schemas.items() if k not in tools_to_hide_this_turn
            }
            current_turn_tool_dispatch = {
                k: v for k, v in available_tool_dispatch.items() if k not in tools_to_hide_this_turn
            }

    return current_turn_tools_schemas, current_turn_tool_dispatch


def advance_to_next_turn(
    session,
    user_turn_index,
    steps_this_user_turn,
    all_user_turns,
    missed_function_config,
    logger,
):
    previous_turn_key = str(user_turn_index + 1)
    user_turn_index += 1
    steps_this_user_turn = 0

    if user_turn_index < len(all_user_turns):
        next_user_query_turn = all_user_turns[user_turn_index]
        logger.info(
            f"Agent finished turn for user query {user_turn_index}. Injecting next user query: {next_user_query_turn}"
        )

        if previous_turn_key in missed_function_config and not next_user_query_turn:
            logger.info(
                f"Injecting special prompt for restored functions: '{DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC}'"
            )
            session.add_to_prompt({"role": "user", "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC})

        session.prompt.extend(next_user_query_turn)
        session.finished = False
    else:
        logger.info("All user turns have been processed. Finishing example.")
        session.finish()

    return user_turn_index, steps_this_user_turn
