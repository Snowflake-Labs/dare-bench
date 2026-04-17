from typing import List, Dict, Any, Set
from agentic_reason.prompt_manager import prepare_active_sequence


class Session:
    """
    A class to manage the state of a single processing session for an example.
    """
    def __init__(
        self,
        example_data: Dict[str, Any],
        config: Dict[str, Any],
        tokenizer_planner: Any,
        available_tool_schemas: Dict[str, Any],
        available_tool_dispatch: Dict[str, Any],
        is_multi_turn: bool,
        logger: Any,
    ):
        self.item: Dict[str, Any] = example_data
        
        # Prepare the initial prompt
        self.prompt: Any = prepare_active_sequence(
            example_data=example_data,
            dataset_name=config.dataset_name,
            model_path=config.llm.planner.model_path,
            tokenizer=tokenizer_planner,
            tool_use_type=config.tool.use_type,
            max_search_limit=config.tool.search.max_search_limit,
            use_fewshot_example=config.llm.planner.use_fewshot_example,
        )
        
        self.output: str = ''
        self.finished: bool = False
        self.history: List[str] = []
        self.search_count: int = 0
        self.executed_search_queries: Set[str] = set()
        self.tools: List[Dict[str, Any]] = list(available_tool_schemas.values()) if available_tool_schemas else []
        self.tools_extra_kwargs: Dict[str, Any] = {}
        self.ground_truth: Any = example_data.get('ground_truth', None)
        self.metadata: Dict[str, Any] = example_data.get('metadata', {}) # contains additional data-specific metadata
        self.task: str = example_data.get('task', None)
        self.num_tool_calls: int = 0

        self._setup_extra_tool_kwargs(config, available_tool_dispatch, is_multi_turn, logger)

    def _setup_extra_tool_kwargs(self, config, available_tool_dispatch, is_multi_turn, logger):
        """Datasci (python_executor) does not need per-dataset analyst/search extras."""
        return

    def finish(self):
        """Marks the session as finished."""
        self.finished = True

    def add_to_prompt(self, message: Dict[str, Any]):
        """Adds a message to the prompt if it's a list."""
        if isinstance(self.prompt, list):
            self.prompt.append(message)

    def pop_from_prompt(self) -> Dict[str, Any] | None:
        """Removes and returns the last message from the prompt if it's a list."""
        if isinstance(self.prompt, list) and self.prompt:
            return self.prompt.pop()
        return None

    def add_to_history(self, text: str):
        """Appends a string to the session's history."""
        self.history.append(text) 