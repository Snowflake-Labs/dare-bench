import argparse
# Special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
BEGIN_CODE_QUERY = "<|begin_code_query|>"
END_CODE_QUERY = "<|end_code_query|>"
BEGIN_CODE_RESULT = "<|begin_code_result|>"
END_CODE_RESULT = "<|end_code_result|>"
BEGIN_MIND_MAP_QUERY = "<|begin_mind_map_query|>"
END_MIND_MAP_QUERY = "<|end_mind_map_query|>"
BEGIN_MIND_MAP_RESULT = "<|begin_mind_map_result|>"
END_MIND_MAP_RESULT = "<|end_mind_map_result|>"
CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ ' ' }}{% endif %}{% endfor %}{{ eos_token }}"

def parse_args():
    parser = argparse.ArgumentParser(description="Run Search O1 for various datasets and models.")

    # Dataset and split configuration
    parser.add_argument(
        '--dataset_name',
        type=str,
        choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'triviaqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle', 'snow'],
        help="Name of the dataset to use."
    )

    parser.add_argument(
        '--split',
        type=str,
        choices=['test', 'diamond', 'main', 'extended'],
        help="Dataset split to use."
    )

    parser.add_argument(
        '--subset_num',
        type=int,
        default=-1,
        help="Number of examples to process. Defaults to all if not specified."
    )

    # Search and document retrieval configuration
    parser.add_argument(
        '--max_search_limit',
        type=int,
        default=10,
        help="Maximum number of searches per question."
    )

    parser.add_argument(
        '--max_turn',
        type=int,
        default=15,
        help="Maximum number of turns."
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help="Maximum number of search documents to return."
    )

    parser.add_argument(
        '--max_doc_len',
        type=int,
        default=3000,
        help="Maximum length of each searched document."
    )

    parser.add_argument(
        '--use_jina',
        type=bool,
        default=True,
        help="Whether to use Jina API for document fetching."
    )

    parser.add_argument(
        '--jina_api_key',
        type=str,
        default='None',
        help="Your Jina API Key to Fetch URL Content."
    )

    # Model configuration
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--model_path',
        type=str,
        help="Path to the pre-trained model. Required if not using a remote model."
    )
    model_group.add_argument(
        '--remote_model',
        type=str,
        choices=['gpt-4o', 'claude-3.5-sonnet'],
        help="Name of remote API-based model to use (OpenAI's GPT-4o or Anthropic's Claude models)."
    )

    # Sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help="Sampling temperature."
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help="Top-p sampling parameter."
    )

    parser.add_argument(
        '--top_k_sampling',
        type=int,
        default=20,
        help="Top-k sampling parameter."
    )

    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=None,
        help="Repetition penalty. If not set, defaults based on the model."
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=32768,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    # Bing API Configuration
    parser.add_argument(
        '--bing_subscription_key',
        type=str,
        default="",
        help="Bing Search API subscription key."
    )

    parser.add_argument(
        '--bing_endpoint',
        type=str,
        default="https://api.bing.microsoft.com/v7.0/search",
        help="Bing Search API endpoint."
    )

    parser.add_argument(
        '--mind_map',
        type=bool,
        default=False,
        help="Whether to use mind map for reasoning."
    )

    parser.add_argument(
        '--mind_map_path',
        type=str,
        default='./local_mem',
        help="Path to the mind map."
    )

    parser.add_argument(
        '--result_path',
        type=str,
        default='./results/dummy',
        help="Output path."
    )

    parser.add_argument(
        '--deep_research',
        type=bool,
        default=False,
        help="Whether to use deep research for reasoning."
    )

    parser.add_argument(
        '--search_backend',
        type=str,
        default='cortexsearch',
        choices=['bing', 'cortexsearch'],
        help="Search engine to use."
    )

    parser.add_argument(
        '--use_fewshot_example',
        type=bool,
        default=False,
        help="Whether to use fewshot examples for reasoning.",
    )

    parser.add_argument(
        '--bird_db_base_path',
        type=str,
        default=None,
        help="Base path to the BIRD database.",
    )

    parser.add_argument(
        '--bird_db_split',
        type=str,
        default="dev",
        choices=["dev", "train"],
        help="Data split to be used for the BIRD database.",
    )

    parser.add_argument(
        '--max_eval_samples',
        type=int,
        default=None,
        help="The number of samples to evaluate.",
    )

    args = parser.parse_args()
    
    # Validate that either model_path or remote_model is provided
    if not args.model_path and not args.remote_model:
        parser.error("Either --model_path or --remote_model must be provided")
    
    return args
