#!/usr/bin/env bash
# Parallel inference driver for datasci benchmarks (extracted from si-tool-agent/Snowmind-generate).
# Run from this directory: ./run_parallel_datasci_auto.sh
# Or from dare-bench/ (this package root): ./scripts/run_parallel_datasci_auto.sh
#
# Unified data layout (default): dare-bench/data/eval/ (next to this scripts/ folder)
#   - question_list.json
#   - databases/<task_folder>/{source,verify}/
#
# Env overrides:
#   DATASCI_OUTPUT_BASE       — output root
#   DATASCI_EVAL_ROOT         — eval bundle root (default: ../data/eval relative to this script)
#   DATASCI_PROBLEM_TYPE      — classification | regression | time_series_analysis
#   CI_RUN_CODE_URL           — python executor URL

fewshot_flag=${2:-0}
analyst_flag=${3:-0}
provide_context_flag=${4:-1}
append_tool_output_flag=${5:-1}
optional_timestamp=${6:-""}
claude="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
claude4="us.anthropic.claude-sonnet-4-20250514-v1:0"
gpt="gpt-4o"
config="datasci"
qwen="Qwen3-32B"
qwen4b="Qwen3-4B"
gpt41="gpt-4-1-dev"
gpt5="gpt-5"
o4mini="gpt-o4-mini"
o3="gpt-o3"
reasoning_effort="high"
reasoning_summary="auto"
ci_url="${CI_RUN_CODE_URL:-http://autoscale-yite:8080/run_code}"

# MAIN MODIFY OPTIONS
model=${gpt}
search="cortexsearch"
mind_map=false
fewshot=false
analyst=${gpt}
hint="true"
output_tag=""
num_runs="1"
max_turn=5
max_eval_samples="null"
force_simulate="true"
timeout=200
comment="MM"
temperature="0.0"
# Hydra dataset split label (matches config/datasci.yaml default)
db_split_dev="dev"

# Single bundle for all problem types; filter by task via DATASCI_PROBLEM_TYPE.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_EVAL_ROOT="${ROOT}/../data/eval"
DATASCI_EVAL_ROOT="${DATASCI_EVAL_ROOT:-$DEFAULT_EVAL_ROOT}"
# classification | regression | time_series_analysis — must match "task" in question_list.json
DATASCI_PROBLEM_TYPE="${DATASCI_PROBLEM_TYPE:-classification}"
DATASCI_PROBLEM_TYPE="classification"

if [ -n "$optional_timestamp" ]; then
  timestamp="$optional_timestamp"
  retry_failed_only=true
else
  timestamp=$(date +"%Y%m%d_%H%M%S")
  retry_failed_only=false
fi

cd "$ROOT"

# Default: dare-bench/output/test (two levels up from scripts/ = package root, then sibling output/)
OUTPUT_BASE="${DATASCI_OUTPUT_BASE:-${ROOT}/../output/test}"

dataset_name="datasci-eval"

set -e
export LOG_LEVEL=WARNING

# loop over question versions (v1 / v2 prompts & ground-truth layout)
for ds_question_version in v2; do
  output_dir="${OUTPUT_BASE}/run_${model}_${dataset_name}_${DATASCI_PROBLEM_TYPE}_${ds_question_version}_${timestamp}_${comment}"
  mkdir -p "$output_dir"
  log_file="$output_dir/run.log"

  echo "=== Running ${dataset_name} / task=${DATASCI_PROBLEM_TYPE} / QV=${ds_question_version} / eval_root=${DATASCI_EVAL_ROOT} ==="

  python run_agentic_reason.py --config-name "$config" \
    dataset_name="$dataset_name" \
    datasci_eval_root="$DATASCI_EVAL_ROOT" \
    datasci_problem_type="$DATASCI_PROBLEM_TYPE" \
    db_split="$db_split_dev" \
    db_base_path="${DATASCI_EVAL_ROOT}/databases" \
    log_path="$output_dir" \
    max_turn="$max_turn" \
    retry_failed_only="$retry_failed_only" \
    llm.planner.use_fewshot_example="$fewshot" \
    llm.extractor.remote_model="$model" \
    llm.planner.remote_model="$model" \
    llm.planner.reasoning_effort=${reasoning_effort} \
    llm.planner.reasoning_summary=${reasoning_summary} \
    llm.planner.temperature=${temperature} \
    tool.analyst.remote_model=${model} \
    tool.analyst.backend=$analyst \
    tool.use_type="function_call" \
    num_workers=96 \
    num_threads=128 \
    num_runs=$num_runs \
    tool.analyst.provide_context=$provide_context \
    tool.analyst.sql_hint=$hint \
    tool.search.backend=$search \
    tool.append_tool_output=$append_tool_output \
    result_path=$output_dir \
    save_chunk_size=1000 \
    max_eval_samples=$max_eval_samples \
    ds_question_version=$ds_question_version \
    force_simulate=$force_simulate \
    tool.python_executor.timeout=$timeout \
    tool.python_executor.url=$ci_url \
    clean_cache=true \
    2>&1 | tee -a "$log_file"

done
