"""Register datasci-only tools (python_executor) for function-calling."""

import inspect
import importlib

from .base import TOOL_REGISTRY, BaseTool


def import_registered_tools():
    import tools.python_executor.python_executor  # noqa: F401 — registers python_executor


def build_function_maps(model, config):
    import_registered_tools()

    FUNCTION_SCHEMAS = {}
    FUNCTION_DISPATCH = {}

    for name, tool in TOOL_REGISTRY.items():
        sig = inspect.signature(tool.get_tool_description)
        if "config" in sig.parameters:
            FUNCTION_SCHEMAS[name] = tool.get_tool_description(model, config)
        else:
            FUNCTION_SCHEMAS[name] = tool.get_tool_description(model)
        FUNCTION_DISPATCH[name] = tool
    return FUNCTION_SCHEMAS, FUNCTION_DISPATCH
