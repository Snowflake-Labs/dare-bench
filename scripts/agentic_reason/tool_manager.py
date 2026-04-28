import logging

from .utils import convert_tool_schema
from tools.loader import build_function_maps


class ToolManager:
    def __init__(self, config, llm_planner, tokenizer_planner, llm_extractor, tokenizer_extractor, logger):
        self.config = config
        self.llm_planner = llm_planner
        self.tokenizer_planner = tokenizer_planner
        self.llm_extractor = llm_extractor
        self.tokenizer_extractor = tokenizer_extractor
        self.logger = logger
        self.TOOL_SCHEMAS, self.TOOL_DISPATCH = build_function_maps(
            model=self.config.llm.planner.remote_model, config=self.config
        )

    def initialize_tools(self, example_data, dataset_name):
        is_multi_turn = example_data.get("multi_turn", False)
        if is_multi_turn:
            return self._initialize_multiturn_tools(example_data, dataset_name)
        return self._initialize_simple_tools(example_data)

    def _initialize_multiturn_tools(self, example_data, dataset_name):
        self.logger.info("Multi-turn scenario detected. Using data-driven tool initialization.")

        data_derived_tool_names = example_data.get("data_derived_tool_names", [])
        original_tool_names = example_data.get("tools", [])
        available_tools_names = list(set(data_derived_tool_names + original_tool_names))
        self.logger.info(f"Tools from data: {available_tools_names}")

        initial_config = example_data.get("initial_config", {})
        self._load_tool_scenarios(available_tools_names, initial_config, dataset_name, example_data)

        final_available_tool_names = []
        heavy_init_cache = set()

        for tool_name in available_tools_names:
            if tool_name not in self.TOOL_DISPATCH:
                self.logger.warning(
                    f"Tool '{tool_name}' specified in data but not found in TOOL_DISPATCH. Skipping."
                )
                continue

            tool_instance = self.TOOL_DISPATCH[tool_name]
            if self._initialize_specific_tool(tool_name, tool_instance, example_data):
                heavy_init_cache.add(tool_name)
                final_available_tool_names.append(tool_name)

        self.logger.info(f"Available tools for this run: {final_available_tool_names}")

        available_tool_schemas = {k: v for k, v in self.TOOL_SCHEMAS.items() if k in final_available_tool_names}
        available_tool_dispatch = {k: v for k, v in self.TOOL_DISPATCH.items() if k in final_available_tool_names}

        return available_tool_schemas, available_tool_dispatch, heavy_init_cache

    def _load_tool_scenarios(self, available_tools_names, initial_config, dataset_name, example_data):
        self.logger.info("Resetting/loading scenarios for stateful tools (if any).")

        example_id = example_data.get("id", "")
        test_category = example_id.rsplit("_", 1)[0]
        use_long_context = "long_context" in test_category or "composite" in test_category
        if use_long_context:
            self.logger.info(
                f"Detected long_context or composite test category ('{test_category}'). Setting long_context=True."
            )

        for tool_name in available_tools_names:
            tool_instance = self.TOOL_DISPATCH.get(tool_name)
            if tool_instance and hasattr(tool_instance, "_load_scenario"):
                scenario_config = initial_config.get(tool_name, {})
                try:
                    tool_instance._load_scenario(scenario_config, long_context=use_long_context)
                    self.logger.info(f"{tool_name} scenario loaded/reset successfully.")
                except Exception as e:
                    self.logger.error(f"Failed to load/reset scenario for {tool_name}: {e}")
            elif tool_instance:
                self.logger.info(f"Tool '{tool_name}' has no '_load_scenario' method; assuming stateless.")

    def _initialize_specific_tool(self, tool_name, tool_instance, example_data):
        try:
            tool_instance.initialize(self.config)
            self.logger.info(f"{tool_name} initialized.")
            return True
        except Exception:
            self.logger.info(f"Tool '{tool_name}': generic initialize path.")
            return True

    def _initialize_simple_tools(self, example_data):
        self.logger.info("Single-turn scenario: static tool schema from example data.")
        raw_tool_schemas = example_data.get("tools", [])
        available_tool_schemas = {}
        if isinstance(raw_tool_schemas, list):
            for raw_tool_schema in raw_tool_schemas:
                if isinstance(raw_tool_schema, dict) and "name" in raw_tool_schema:
                    available_tool_schemas[raw_tool_schema["name"]] = convert_tool_schema(
                        raw_tool_schema, model=self.config.llm.planner.remote_model
                    )
                else:
                    self.logger.warning(f"Skipping malformed tool schema: {raw_tool_schema}")
        else:
            self.logger.warning(f"Expected 'tools' to be a list of schemas, got: {type(raw_tool_schemas)}.")

        available_tool_dispatch = {}
        heavy_init_cache = set()

        return available_tool_schemas, available_tool_dispatch, heavy_init_cache

    def cleanup(self, tool_dispatch, dataset_name):
        pass
