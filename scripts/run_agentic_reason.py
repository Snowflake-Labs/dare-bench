import json
import os
import shutil
import time

import hydra
from omegaconf import OmegaConf
from tenacity import RetryError

from utils import *
from agentic_reason.generation import run_generation
from agentic_reason.models import initialize_model
from agentic_reason.tool_manager import ToolManager
from agentic_reason.session import Session
from agentic_reason.core_flow import get_current_turn_tools, advance_to_next_turn, setup_dynamic_tool_loading
from agentic_reason.utils import extract_tool_calls_fc


def process_example(example_data, dataset_name, config, run_id=0):
    start_time = time.time()

    llm_planner, tokenizer_planner = initialize_model(config.llm.planner)

    llm_extractor, tokenizer_extractor = None, None
    if (
        hasattr(config, "tool")
        and hasattr(config.tool, "search")
        and hasattr(config.tool.search, "extractor")
        and config.tool.search.extractor.get("enabled", True)
    ):
        llm_extractor, tokenizer_extractor = initialize_model(config.tool.search.extractor)

    example_id = example_data["id"]
    if config.skip_filtered_data and example_data.get("is_filtered", False):
        print(f"Skipping filtered question {example_id}")
        return {
            "id": example_id,
            "result": [],
            "run_id": run_id,
            "full_output": "Skipped due to content filter",
            "success": True,
            "messages": [],
            "tools": [],
            "skipped_due_to_filter": True,
        }

    sub_category = dataset_name.split("-")[1]
    model_suffix = "-FC" if config.tool.use_type == "function_call" else ""
    raw_id = example_id[len(sub_category) + 1 :]
    final_id = raw_id.zfill(4)
    if config.log_path:
        example_dir = os.path.join(
            config.log_path, f"{config.llm.planner.remote_model}{model_suffix}", dataset_name, final_id
        )
    else:
        example_dir = os.path.join(config.result_path, final_id)
    run_dir = os.path.join(example_dir, f"run_{run_id:03d}")
    os.makedirs(run_dir, exist_ok=True)

    logger_path = os.path.join(run_dir, "run.log")
    logger = setup_logger(logger_path, config)
    logger.info(f"Processing example {example_id}, run {run_id}")

    OmegaConf.set_struct(config, False)

    cache_dir = os.path.join(run_dir, "cache")
    output_dir = os.path.join(run_dir, "output")
    config = OmegaConf.merge(config, {"cache_dir": cache_dir, "output_dir": output_dir})
    OmegaConf.set_struct(config, True)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)
    os.makedirs(output_dir, exist_ok=True)

    if "original_file_path" in example_data:
        original_file_path = example_data["original_file_path"]
        assert "needed_files" in example_data["metadata"], (
            f"'needed_files' key not found in example_data for {example_id}"
        )
        for file_name in example_data["metadata"]["needed_files"]:
            src_path = os.path.join(original_file_path, file_name)
            assert os.path.exists(src_path), f"Needed file {file_name} not found in {original_file_path}"
            dst_path = os.path.join(cache_dir, file_name)
            shutil.copy2(src_path, dst_path)
            logger.info(f"Copied needed file {file_name} to cache directory for example {example_id}")

    is_multi_turn = example_data.get("multi_turn", False)

    tool_manager = ToolManager(
        config,
        llm_planner,
        tokenizer_planner,
        llm_extractor,
        tokenizer_extractor,
        logger,
    )
    available_tool_schemas, available_tool_dispatch, heavy_init_cache = tool_manager.initialize_tools(
        example_data, dataset_name
    )

    session = Session(
        example_data=example_data,
        config=config,
        tokenizer_planner=tokenizer_planner,
        available_tool_schemas=available_tool_schemas,
        available_tool_dispatch=available_tool_dispatch,
        is_multi_turn=is_multi_turn,
        logger=logger,
    )

    try:
        start_time = time.time()

        batch_output_records = []
        turn = 0
        user_turn_index = 0
        all_user_turns = example_data.get("question", []) if is_multi_turn else []
        logger.info(f"all_user_turns: {all_user_turns}")

        missed_function_config, action_to_tool_map = setup_dynamic_tool_loading(example_data, is_multi_turn, logger)

        if is_multi_turn and all_user_turns:
            per_turn_max_steps = config.max_turn
            total_max_turns = per_turn_max_steps * len(all_user_turns)
            logger.info(
                f"Multi-turn scenario: per-turn step limit={per_turn_max_steps}, total step limit={total_max_turns}."
            )
        else:
            total_max_turns = config.max_turn
            per_turn_max_steps = config.max_turn
            logger.info(f"Single-turn/simple scenario: total step limit={total_max_turns}.")

        steps_this_user_turn = 0

        initial_prompt_content = ""
        if session.prompt:
            if isinstance(session.prompt, list):
                initial_prompt_content = session.prompt[-1].get("content", "")
            else:
                initial_prompt_content = session.prompt
        logger.info(
            f"\n-------------- Example {example_id}, Prompt to Start --------------\n" + initial_prompt_content
        )

        while not session.finished:
            if turn >= total_max_turns:
                logger.info(f"Maximum number of total agent turns ({total_max_turns}) reached, stopping.")
                break

            turn += 1
            steps_this_user_turn += 1

            current_turn_tools_schemas, current_turn_tool_dispatch = get_current_turn_tools(
                available_tool_schemas,
                available_tool_dispatch,
                missed_function_config,
                action_to_tool_map,
                user_turn_index,
                logger,
            )

            logger.info(
                f"\n-------------- Example {example_id}, Run {run_id}, Agent Turn {turn} "
                f"(Step {steps_this_user_turn} for user turn {user_turn_index + 1}) --------------\n"
                f"Beginning generation..."
            )

            made_tool_call_this_turn = False

            outputs, _ = run_generation(
                session,
                llm_planner,
                tokenizer_planner,
                config.llm.planner,
                tools=current_turn_tools_schemas,
                add_tool_hint=config.get("add_tool_hint", True),
                logger=logger,
            )
            logger.info("Generation completed, processing outputs...")

            output = outputs[0]
            tool_call_made = llm_planner.process_messages(
                session,
                output,
                config,
                logger,
                current_turn_tool_dispatch,
                current_turn_tools_schemas,
                heavy_init_cache,
                is_multi_turn,
            )
            if tool_call_made:
                made_tool_call_this_turn = True

            if is_multi_turn:
                force_next_turn = steps_this_user_turn >= per_turn_max_steps

                if not made_tool_call_this_turn or force_next_turn:
                    if force_next_turn and made_tool_call_this_turn:
                        logger.warning(
                            f"Exceeded max steps ({per_turn_max_steps}) for user turn {user_turn_index+1}. Forcing next turn."
                        )

                    user_turn_index, steps_this_user_turn = advance_to_next_turn(
                        session,
                        user_turn_index,
                        steps_this_user_turn,
                        all_user_turns,
                        missed_function_config,
                        logger,
                    )
            else:
                logger.info("Single-turn scenario: model has responded. Finishing example.")
                session.finish()

        total_time = time.time() - start_time
        t = time.localtime()
        batch_output_file = os.path.join(output_dir, f"dev.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.info_extract.json")
        with open(batch_output_file, "w", encoding="utf-8") as f:
            json.dump(batch_output_records, f, ensure_ascii=False, indent=2)
        logger.info(f"Elapsed time: {total_time} seconds\n" + f"Batch outputs saved to {batch_output_file}")

        final_history = session.prompt

        if is_multi_turn:
            logger.info("Formatting results for multi-turn evaluation.")
            final_result_for_eval = get_final_result_for_eval(session, config, None)
        else:
            logger.info("Formatting results using extract_tool_calls_fc.")
            final_result_for_eval = extract_tool_calls_fc(final_history)

        output_list = [session.output]
        full_output_list = ["".join(session.history)]

        final_output_file = os.path.join(run_dir, "final_output.json")

        with open(final_output_file, "w", encoding="utf-8") as f:
            if not config.dataset_name.startswith("datasci"):
                json.dump(
                    {"full_output": full_output_list[0], "messages": session.prompt, "tools": session.tools},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        return {
            "id": example_id,
            "result": final_result_for_eval,
            "run_id": run_id,
            "full_output": full_output_list[0] if full_output_list else None,
            "success": True,
            "batch_output_file": batch_output_file,
            "final_output_file": final_output_file,
            "messages": session.prompt,
            "tools": session.tools,
            "skipped_due_to_filter": False,
            "total_time": time.time() - start_time,
        }

    except RetryError as e:
        import traceback

        tb = traceback.format_exc()
        e_ = e.last_attempt.exception()
        logger.error(
            "\n### [Final Response] ###\n"
            + f"RetryError ENCOUNTERED DURING GENERATION; ORIGINAL ERROR: {e_}\n"
            + f"{tb}\n"
            + "### [/Final Response] ###\n"
        )
        return {
            "id": example_id,
            "run_id": run_id,
            "output": None,
            "full_output": None,
            "success": False,
            "error": str(e_),
            "messages": session.prompt if "session" in locals() else [],
            "tools": session.tools if "session" in locals() else [],
            "traceback": str(tb),
            "result": [],
            "skipped_due_to_filter": False,
            "total_time": time.time() - start_time,
        }

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        logger.error(
            "\n### [Final Response] ###\n" + f"ERROR ENCOUNTERED DURING GENERATION, {e}\n" + f"{tb}\n" + "### [/Final Response] ###\n"
        )
        return {
            "id": example_id,
            "run_id": run_id,
            "output": None,
            "full_output": None,
            "messages": session.prompt if "session" in locals() else [],
            "tools": session.tools if "session" in locals() else [],
            "success": False,
            "error": str(e),
            "traceback": str(tb),
            "result": [],
            "skipped_due_to_filter": False,
            "total_time": time.time() - start_time,
        }

    finally:
        if config.get("clean_cache", False):
            shutil.rmtree(cache_dir)
        tool_manager.cleanup(available_tool_dispatch, dataset_name)
        cleanup_logger(logger)


@hydra.main(config_path="config", config_name="datasci", version_base=None)
def main(config):
    print(OmegaConf.to_yaml(config, resolve=True, sort_keys=False))

    if getattr(config, "sequential", False):
        print("Running in SEQUENTIAL mode.")
        from agentic_reason.execution_manager import main_sequential

        main_sequential(config)
    elif getattr(config, "ordered_parallel", False):
        print("Running in ORDERED PARALLEL mode.")
        from agentic_reason.execution_manager import ordered_parallel

        ordered_parallel(config)
    else:
        print("Running in PARALLEL mode.")
        from agentic_reason.execution_manager import main_parallel

        main_parallel(config)


if __name__ == "__main__":
    main()
