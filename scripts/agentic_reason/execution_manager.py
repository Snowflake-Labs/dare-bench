"""
Execution Manager for handling parallel processing, early stopping, and continue generation logic.

This module contains the logic for:
- Checking existing outputs and determining if examples should be skipped
- Managing parallel execution with early stopping
- Handling continuation from partial runs
"""

import os
import json
import multiprocessing
from collections import defaultdict
import time
import errno
import sys
import logging
import concurrent.futures
from tqdm import tqdm
import glob

from utils import setup_execution_logger, get_processed_data_dict

# Handle fcntl import for cross-platform compatibility
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # fcntl is not available on Windows
    HAS_FCNTL = False


# Global logger instance
_execution_logger = None


def get_execution_logger(config=None):
    """Get or create the execution logger instance."""
    global _execution_logger
    if _execution_logger is None:
        
        if config is not None:
            _execution_logger = setup_execution_logger(config)
        else:
            # Create a default console-only logger if no config provided
            _execution_logger = logging.getLogger("ExecutionManager")
            _execution_logger.setLevel(logging.INFO)
            if not _execution_logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                _execution_logger.addHandler(handler)
    return _execution_logger


def _run_with_process_example_wrapper(args):
    """
    Module-level wrapper function for multiprocessing compatibility.
    This function can be pickled unlike local functions defined inside other functions.
    """
    # Import here to avoid circular imports
    from run_agentic_reason import process_example
    
    return run_all_runs_for_example(args, process_example)


def acquire_example_lock(example_dir, example_id, timeout=300):
    """
    Acquire a file-based lock for processing an example to prevent race conditions.
    
    Args:
        example_dir: Directory path for the example
        example_id: Unique identifier for the example
        timeout: Maximum time to wait for lock (seconds)
        
    Returns:
        tuple: (lock_file_handle, success) where success is True if lock acquired
    """
    if not HAS_FCNTL:
        # On Windows or systems without fcntl, use a simple file existence check
        # This is a basic fallback that provides some protection but not bulletproof
        os.makedirs(example_dir, exist_ok=True)
        lock_file_path = os.path.join(example_dir, f".processing_lock_{example_id}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to create lock file exclusively
                lock_file = open(lock_file_path, 'x')  # 'x' mode fails if file exists
                lock_file.write(f"pid:{os.getpid()},time:{time.time()}\n")
                lock_file.flush()
                return lock_file, True
            except FileExistsError:
                # Lock file exists, wait and retry
                time.sleep(1)
                continue
            except Exception:
                return None, False
        
        return None, False
    
    # Unix-like systems with fcntl support
    os.makedirs(example_dir, exist_ok=True)
    lock_file_path = os.path.join(example_dir, f".processing_lock_{example_id}")
    
    try:
        lock_file = open(lock_file_path, 'w')
        
        # Try to acquire exclusive lock with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Successfully acquired lock - write our process info
                lock_file.write(f"pid:{os.getpid()},time:{time.time()}\n")
                lock_file.flush()
                return lock_file, True
            except IOError as e:
                if e.errno == errno.EAGAIN or e.errno == errno.EACCES:
                    # Lock is held by another process, wait and retry
                    time.sleep(1)
                    continue
                else:
                    # Other error
                    lock_file.close()
                    return None, False
        
        # Timeout exceeded
        lock_file.close()
        return None, False
        
    except Exception as e:
        return None, False


def release_example_lock(lock_file, example_dir, example_id):
    """
    Release the file-based lock for an example.
    
    Args:
        lock_file: File handle returned by acquire_example_lock
        example_dir: Directory path for the example
        example_id: Unique identifier for the example
    """
    if lock_file:
        try:
            if HAS_FCNTL:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            
            # Clean up lock file
            lock_file_path = os.path.join(example_dir, f".processing_lock_{example_id}")
            if os.path.exists(lock_file_path):
                os.remove(lock_file_path)
        except Exception:
            pass  # Best effort cleanup


def check_existing_outputs(example_data, dataset_name, config, num_runs=1):
    """Continue-from-partial-run was only implemented for analytics; disabled for datasci."""
    return False, 0, "datasci build: continue_generation not implemented"


def get_example_dir(example_data, dataset_name, config):
    """Helper function to get the example directory path"""
    example_id = example_data["id"]
    sub_category = dataset_name.split("-")[1]
    model_suffix = "-FC" if config.tool.use_type == "function_call" else ""
    raw_id = example_id[len(sub_category) + 1:]
    final_id = raw_id.zfill(4)
    
    if config.log_path:
        return os.path.join(config.log_path, f"{config.llm.planner.remote_model}{model_suffix}", dataset_name, final_id)
    else:
        return os.path.join(config.result_path, final_id)


def run_all_runs_for_example(args, process_example_func=None):
    """
    Execute all runs for a single example with early stopping and continuation support.
    
    Args:
        args: Tuple of (dataset_name, example_data, config, num_runs)
        process_example_func: Function to process a single example (to avoid circular imports)
        
    Returns:
        Tuple of (dataset_name, results)
    """
    if process_example_func is None:
        # Import here to avoid circular imports
        from run_agentic_reason import process_example
        process_example_func = process_example
    
    dataset_name, example_data, config, num_runs = args
    example_id = example_data["id"]
    logger = get_execution_logger(config)
    
    # Get the example directory and try to acquire lock
    example_dir = get_example_dir(example_data, dataset_name, config)
    lock_file, lock_acquired = acquire_example_lock(example_dir, example_id, timeout=10)
    
    if not lock_acquired:
        logger.info(f"[{dataset_name}] Could not acquire lock for example {example_id} (another worker is processing it)")
        return dataset_name, []  # Return empty results for locked examples
    
    try:
        # Check if we should skip this example due to existing outputs
        should_skip, existing_completed_runs, reason = check_existing_outputs(example_data, dataset_name, config, num_runs)
        
        if should_skip:
            logger.info(f"[{dataset_name}] Skipping example {example_id}: {reason}")
            return dataset_name, []  # Return empty results for skipped examples
        elif existing_completed_runs > 0:
            logger.info(f"[{dataset_name}] Continuing example {example_id}: {reason}")
        
        seen_eval_1 = False
        seen_eval_0 = False
        first_eval_1_run_id = None
        results = []
        
        # If we have existing runs, analyze them to continue from the right point
        if existing_completed_runs > 0:
            # Load existing evaluations to initialize early stop state
            for existing_run_id in range(existing_completed_runs):
                run_dir = os.path.join(example_dir, f"run_{existing_run_id:03d}")
                final_output_file = os.path.join(run_dir, "final_output.json")
                if os.path.exists(final_output_file):
                    try:
                        with open(final_output_file, 'r') as f:
                            result_data = json.load(f)
                        eval_result = result_data.get("evaluation", None)
                        if eval_result == 1:
                            seen_eval_1 = True
                            if first_eval_1_run_id is None:
                                first_eval_1_run_id = existing_run_id
                        elif eval_result == 0:
                            seen_eval_0 = True
                    except:
                        continue

        # Start from where we left off
        start_run_id = existing_completed_runs
        
        for run_id in range(start_run_id, num_runs):
            # Early stop conditions when config.early_stop is True:
            # 1. We have seen both eval 0 and eval 1
            # 2. We have completed all runs
            if config.early_stop and seen_eval_1 and seen_eval_0:
                logger.info(f"[{dataset_name}] Early stopped example {example_id} after seeing both 0 and 1.")
                break
            try:
                result = process_example_func(example_data, dataset_name, config, run_id)
                eval_result = result.get("evaluation")
                if eval_result == 1:
                    seen_eval_1 = True
                    if first_eval_1_run_id is None:
                        first_eval_1_run_id = run_id
                elif eval_result == 0:
                    seen_eval_0 = True

                result["early_stop_triggered"] = seen_eval_1 and seen_eval_0
                result["first_eval_1_run_id"] = first_eval_1_run_id
                results.append(result)
                logger.info(f"[{dataset_name}] ✓ Completed example {example_id}, run {run_id}")
            except Exception as e:
                logger.error(f"[{dataset_name}] ✗ Exception in example {example_id}, run {run_id}: {e}")
                results.append({
                    "id": example_id,
                    "run_id": run_id,
                    "output": None,
                    "success": False,
                    "error": str(e),
                    "early_stop_triggered": False
                })
        
        # Check if we completed all runs without early stopping
        if config.early_stop and start_run_id + len(results) >= num_runs and not (seen_eval_1 and seen_eval_0):
            logger.info(f"[{dataset_name}] Completed all {num_runs} runs for example {example_id} without early stopping.")
        elif not config.early_stop and start_run_id + len(results) >= num_runs:
            logger.info(f"[{dataset_name}] Completed all {num_runs} runs for example {example_id}.")

        return dataset_name, results
    
    finally:
        # Always release the lock when done
        release_example_lock(lock_file, example_dir, example_id)


def ordered_parallel(config):
    """
    Execute examples in ordered parallel mode with early stopping and continuation support.
    
    Args:
        config: Configuration object containing execution parameters
    """
    # Import here to avoid circular imports
    
    processed_data_dict = get_processed_data_dict(config)
    num_runs = getattr(config, 'num_runs', 8)
    num_workers = getattr(config, 'num_workers', 8)

    total_examples = sum(len(v) for v in processed_data_dict.values())
    logger = get_execution_logger(config)
    logger.info(f"Starting ordered-parallel execution with {num_workers} workers for {len(processed_data_dict)} datasets, "
               f"{total_examples} total examples, {num_runs} runs each")

    # Flatten into a list of (dataset_name, example_data, config, num_runs)
    dataset_example_args = []
    for dataset_name, processed_data in processed_data_dict.items():
        for filtered_data in processed_data:
            dataset_example_args.append((dataset_name, filtered_data[0], config, num_runs))

    # Use multiprocessing for parallelism across examples
    with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
        dataset_results = pool.map(_run_with_process_example_wrapper, dataset_example_args)
    # Note: Results are not saved in this function - handled by the caller if needed

    logger.info("All examples processed.")


def main_parallel(config):
    """
    Execute examples in main parallel mode with chunking and result saving.
    
    Args:
        config: Configuration object containing execution parameters
    """
    # Import here to avoid circular imports
    
    # Import process_example function
    from run_agentic_reason import process_example

    processed_data_dict = get_processed_data_dict(config)
    num_runs = config.num_runs if hasattr(config, 'num_runs') else 8
    num_workers = config.num_workers if hasattr(config, 'num_workers') else 8
    num_threads = config.num_threads if hasattr(config, 'num_threads') else 64

    total_examples = sum(len(processed_data) for processed_data in processed_data_dict.values())
    logger = get_execution_logger(config)
    logger.info(f"Starting parallel execution with {num_workers} workers for {len(processed_data_dict)} datasets, "
               f"{total_examples} total examples, {num_runs} runs each")

    if config.retry_failed_only:
        # Only re-run failed cases due to some exceptions, mainly rate limit exceeded
        tasks = []
        for dataset_name, processed_data in processed_data_dict.items():
            dataset_result_path = os.path.join(config.result_path, dataset_name)
            if not os.path.exists(dataset_result_path):
                continue
                
            for example_idx_dir in sorted(os.listdir(dataset_result_path)):
                example_path = os.path.join(dataset_result_path, example_idx_dir)
                if not os.path.isdir(example_path):
                    continue
                for run_dir in sorted(os.listdir(example_path)):
                    if not run_dir.startswith("run_"):
                        continue
                    run_path = os.path.join(example_path, run_dir)
                    final_output_file = os.path.join(run_path, "final_output.json")
                    if not os.path.exists(final_output_file):
                        # this should be run again
                        example_idx = int(example_idx_dir)
                        run_id = int(run_dir.split("_")[-1])
                        tasks.append((dataset_name, example_idx, processed_data[example_idx], config, run_id))
    else:
        # Create tasks for all datasets, examples and runs
        # Interleave tasks from different datasets to maximize efficiency
        tasks = []
        
        # First, collect all (dataset, example) pairs
        dataset_example_pairs = []
        for dataset_name, processed_data in processed_data_dict.items():
            for filtered_data in processed_data:
                dataset_example_pairs.append((dataset_name, filtered_data[0]))
        
        # Then create tasks in run-wise order (like original code)
        # but now including dataset information
        for run_id in range(num_runs):
            for dataset_name, example_data in dataset_example_pairs:
                tasks.append((example_data, dataset_name, config, run_id))

    # Execute tasks in parallel
    # Group results by dataset for separate file writing
    results_by_dataset_eval = {dataset_name: [] for dataset_name in processed_data_dict.keys()}
    results_by_dataset_chunk = {dataset_name: [] for dataset_name in processed_data_dict.keys()}
    chunk_idx_by_dataset = {dataset_name: 0 for dataset_name in processed_data_dict.keys()}

    # prepare the stamps
    rerun_stamp = "_rerun" if config.retry_failed_only else ""
    model_suffix = "-FC" if config.tool.use_type == "function_call" else ""
    result_dir = os.path.join(config.result_path, f"{config.llm.planner.remote_model}{model_suffix}")
    os.makedirs(result_dir, exist_ok=True)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_example, *task): task
            for task in tasks
        }

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(future_to_task)):
            example_data, dataset_name, __, run_id = future_to_task[future]
            example_id = example_data['id']
            
            try:
                result = future.result()
                results_by_dataset_eval[dataset_name].append(result["result"] if result["result"] else {})
                results_by_dataset_chunk[dataset_name].append(result)
                if len(results_by_dataset_chunk[dataset_name]) >= config.get("save_chunk_size", 1000):
                    chunk_fname = os.path.join(
                        result_dir,
                        f"parallel_results_{dataset_name}_chunk_{chunk_idx_by_dataset[dataset_name]:03d}_{rerun_stamp}.json"
                    )
                    with open(chunk_fname, "w", encoding="utf-8") as cf:
                        json.dump(results_by_dataset_chunk[dataset_name], cf, indent=2, ensure_ascii=False)
                    results_by_dataset_chunk[dataset_name] = []
                    chunk_idx_by_dataset[dataset_name] += 1
                    logger.info(f"–– flushed chunk {chunk_idx_by_dataset[dataset_name]}, {len(results_by_dataset_chunk[dataset_name])} results → {chunk_fname}")
                logger.info(f"✓ Completed dataset {dataset_name}, example {example_id}, run {run_id + 1}/{num_runs}")
            except Exception as exc:
                logger.error(f"✗ Dataset {dataset_name}, example {example_id}, run {run_id + 1} generated an exception: {exc}")

    # Save results for each dataset separately
    for dataset_name, results in results_by_dataset_chunk.items():
        # flush any remaining results
        if results:
            chunk_fname = os.path.join(
                result_dir,
                f"parallel_results_{dataset_name}_chunk_{chunk_idx_by_dataset[dataset_name]:03d}.json"
            )
            with open(chunk_fname, "w", encoding="utf-8") as cf:
                json.dump(results, cf, indent=2, ensure_ascii=False)
            logger.info(f"–– flushed final chunk {chunk_idx_by_dataset[dataset_name]}, {len(results)} results → {chunk_fname}")
    
    # --- Merge chunks into final result files ---
    logger.info("Merging chunk files into final result files...")
    for dataset_name in processed_data_dict.keys():
        all_results = []
        chunk_files = sorted(glob.glob(os.path.join(result_dir, f"parallel_results_{dataset_name}_chunk_*.json")))
        
        if not chunk_files:
            logger.warning(f"No chunk files found for dataset '{dataset_name}'. Skipping merge.")
            continue

        logger.info(f"Found {len(chunk_files)} chunk files for '{dataset_name}'. Merging...")
        for chunk_file in chunk_files:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                all_results.extend(json.load(f))
        
        # Write the final merged file
        final_result_file = os.path.join(result_dir, f"{dataset_name}.json")
        with open(final_result_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Merged {len(all_results)} results into '{final_result_file}'")

        # Clean up the chunk files
        for chunk_file in chunk_files:
            os.remove(chunk_file)
        logger.info(f"  - Cleaned up {len(chunk_files)} chunk files for '{dataset_name}'.")

    # --- End of merge logic ---
    if config.dataset_name.startswith("datasci"):
        for dataset_name, results in results_by_dataset_eval.items():
            prediction_exist = [r.get("prediction_exist", 0.0) for r in results]
            final_scores = [r.get("final_score", 0.0) for r in results]
            print(f"prediction exist for dataset {dataset_name}: {sum(prediction_exist)}/{len(results)} = {sum(prediction_exist)/len(results)} ")
            print(f"Accuracy for dataset {dataset_name}: {sum(final_scores)}/{len(results)} = {sum(final_scores) / len(results)} among {len(results)} examples.")

    logger.info(f"All tasks completed for {len(processed_data_dict)} datasets")


def main_sequential(config):
    """
    Execute examples in sequential mode with continue generation support.
    
    Args:
        config: Configuration object containing execution parameters
    """
    # Import here to avoid circular imports
    from run_agentic_reason import process_example
    
    # Correctly call get_processed_data with both arguments
    processed_data_dict = get_processed_data_dict(config)

    logger = get_execution_logger(config)
    logger.info(f"Starting sequential execution for {len(processed_data_dict)} examples, one run each")

    results = []
    skipped_count = 0

    for dataset_name, processed_data in processed_data_dict.items():
        for filtered_data in processed_data:
            example_data = filtered_data[0]
            
            # Check if we should skip this example due to existing outputs
            should_skip, completed_runs, reason = check_existing_outputs(example_data, dataset_name, config, num_runs=1)
            
            if should_skip:
                logger.info(f"Skipping example {example_data['id']}: {reason}")
                skipped_count += 1
                continue
            elif completed_runs > 0:
                logger.info(f"Continuing example {example_data['id']}: {reason}")
            
            # Correctly call process_example with the dataset_name
            result = process_example(example_data, dataset_name, config, run_id=0)
            results.append(result)
            logger.info(f"✓ Completed example {example_data['id']}")

    # Save the overall results
    result_fname = f"{config.dataset_name}.json"
    model_suffix = "-FC" if config.tool.use_type == "function_call" else ""
    result_dir = os.path.join(config.result_path, f"{config.llm.planner.remote_model}{model_suffix}")
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, result_fname)
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"All tasks completed. Results saved to {result_file}")
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} examples with existing outputs")