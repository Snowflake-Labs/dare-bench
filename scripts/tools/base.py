from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
# Global registry populated on subclass definition
TOOL_REGISTRY: dict[str, "BaseTool"] = {}

def register_tool(cls):
    """
    Decorator to instantiate and register a concrete tool class.
    """
    inst = cls(config={})
    TOOL_REGISTRY[inst.name] = inst
    return cls

class BaseTool(ABC):
    """
    Every tool must subclass this. On subclass definition we instantiate
    it and stick it into TOOL_REGISTRY[name].
    """
    name: str               # unique tool identifier
    description: str

    def __init__(self, config: Dict[str, Any]):
        """
        Every tool gets a config dict at construction.
        Concrete classes should set abstract=False, define name, description.
        """
        self.config      = config
        self.initialized = False

    def initialize(self):
        """
        Default initializer: no-op beyond marking initialized.
        Subclasses that need heavy setup can override but should
        call super().initialize() at the end.
        """
        self.initialized = True

    @abstractmethod
    def __call__(self, **kwargs) -> Tuple[str, bool]:
        """Execute the tool’s logic and return (result, if successful)."""
        ...

    @abstractmethod
    def get_tool_description(self) -> dict:
        """
        Return a dict:
        {
          "name": ...,
          "description": ...,
          "parameters": { type:"object", properties:{...}, required:[...] }
        }
        """
        ...
