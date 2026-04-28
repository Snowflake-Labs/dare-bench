import os
import requests
import json
from typing import List, Optional, Iterable, Any, Union, Dict, Set, Tuple
from tools.base import BaseTool
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from agentic_reason.config import (
    BEGIN_SEARCH_QUERY,
    END_SEARCH_QUERY,
    BEGIN_CODE_QUERY,
    END_CODE_QUERY,
    BEGIN_SEARCH_RESULT,
    END_SEARCH_RESULT,
    BEGIN_CODE_RESULT,
    END_CODE_RESULT,
    BEGIN_MIND_MAP_QUERY,
    END_MIND_MAP_QUERY,
    BEGIN_MIND_MAP_RESULT,
    END_MIND_MAP_RESULT,
)

from agentic_reason.utils import (
    extract_between,
    extract_reasoning_context,
)
from agentic_reason.session import Session

from collections import defaultdict


def _openai_provider_from_config(config) -> str:
    """
    Which OpenAI-compatible backend to use: 'azure' (Azure OpenAI) or 'openai' (api.openai.com).
    Hydra: llm.planner.openai_provider / extractor.openai_provider, or env OPENAI_PROVIDER (default azure).
    """
    if config is not None and hasattr(config, "get"):
        p = config.get("openai_provider")
        if p is not None and str(p).strip():
            v = str(p).lower().strip()
            if v in ("azure", "openai"):
                return v
    env = os.environ.get("OPENAI_PROVIDER", "azure").lower().strip()
    if env in ("azure", "openai"):
        return env
    return "azure"


@dataclass
class SamplingParams:
    max_tokens: int = 2000
    max_think_tokens: int = 1000
    temperature: float = 0.001
    top_p: float = 0.8
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop: list = field(default_factory=list)
    include_stop_str_in_output: bool = True
    
class VLLMDeploymentAPI:
    def __init__(
            self,
            model_name: str,
            https_end_point: str,
    ):
        self.model_name = model_name
        self.https_end_point = https_end_point
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join("/data/boyiliu/models/", model_name))
        
        
    def create_completion(
            self,
            model: str,
            messages: Iterable[Any],
            max_tokens: int = 4096,
            temperature: float = 0.001,
            top_p: float = 0.8,
            top_k: int = 20,
            repetition_penalty: float = 1.0,
            stop: List[str] = [],
            include_stop_str_in_output: bool = True,
    ):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        payload = {
            "model": model,
            "prompt": text,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "stop": stop,
            "include_stop_str_in_output": include_stop_str_in_output,
            "stream": False
        }
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(self.https_end_point, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"VLLM https request failed, status code: {response.status_code}\n")
            print(f"response:\n {response.json()}")
            return f"Generation failed for model {model}"

class BaseRemoteAPILLM(ABC):
    def __init__(self, model_name: str, config = None):
        self.model_name = model_name
        self._init_client(config)   # hook for subclasses

    @abstractmethod
    def _init_client(self, config):
        """Initialize self.client (or self.anthropic / self.qwen)"""
        ...

    @abstractmethod
    def generate(self, prompts: List[Union[str, List[dict]]], sampling_params: Optional[SamplingParams] = None, tools: Optional[List[dict]] = None, **kwargs) -> List:
        """
        Generate responses for a list of prompts using the specified model.
        for Anthropic, the prompts should be a list of dictionaries with 'role' and 'content' keys.
        """
        ...

    @abstractmethod
    def invoke(self, prompt: str) -> type:
        """Method for code generation compatibility"""
        ...

    @abstractmethod
    def process_messages(self, seq, out, config, logger, mind_map_path, search_tool, code_tool,batch_processor, sql_hint, semantic_model, is_multi_turn) -> str:
        """
        Process messages from the model output and update the sequence.
        This method is responsible for handling tool calls and updating the sequence state.
        """
        ...

class OpenAIChatCompletionClient(BaseRemoteAPILLM):
    def _init_client(self, config):
        provider = _openai_provider_from_config(config)
        self.openai_provider = provider

        if provider == "azure":
            # Deployment name used on this Azure resource (legacy default).
            if self.model_name == "gpt-4o":
                self.model_name = "gpt-4o-rag-research"
            from openai import AzureOpenAI
            self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Azure OpenAI: set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and OPENAI_API_VERSION."
                )
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=os.environ.get("OPENAI_API_VERSION"),
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            )
        else:
            from openai import OpenAI
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OpenAI API (api.openai.com): set OPENAI_API_KEY, or use openai_provider=azure for Azure."
                )
            kwargs = {"api_key": self.api_key}
            base_url = os.environ.get("OPENAI_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
            self.client = OpenAI(**kwargs)

    def generate(self, prompts: List[Union[str, List[dict]]], sampling_params: Optional[SamplingParams] = None, tools: Optional[List[dict]] = None, **kwargs) -> List:
        if sampling_params is None:
            sampling_params = SamplingParams()
        responses = []
        raw_text = []
        for prompt in prompts:
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt         
            request_args = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "stop": sampling_params.stop,
                "n": 1,
            }
            if tools is not None:
                request_args["tools"] = list(tools.values())
                # request_args["parallel_tool_calls"] = False         
            response = self.client.chat.completions.create(**request_args)

        # Note that response contains finish_reason, and messages
        responses.append(response.choices[0])
        raw_text.append(response.choices[0].message.content)
        return responses, raw_text
    
    def invoke(self, prompt: str) -> type:
        """Method for code generation compatibility"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.001,
            max_tokens=2000,
        )
        return type("Response", (), {"content": response.choices[0].message.content})

    def log_response(self, out, logger=None):
        """Pretty print a message response with thinking blocks."""
        # logger.info("\n==== FULL RESPONSE ====")

        # deal with text
        text, thinking, tool_use = out.message.content if out.message.content is not None else "", "", ""
        if logger:
            logger.info("\nANSWER:\n" + text)
        
        # deal with tool_use
        if out.finish_reason == 'tool_calls':
            tool_calls = out.message.tool_calls
            for tool_call in tool_calls:
                if logger:
                    logger.info(f"\nTOOL USE: {tool_call.function.name}\n" + tool_call.function.arguments)
                tool_use += (tool_call.function.name + "\n" + tool_call.function.arguments)
        
        return text, thinking, tool_use

    def dispatch_tool_call(
        self,
        call,
        session: Session,
        dispatch_map: Dict[str, BaseTool],
        heavy_init: Set[str],
    ) -> Tuple[str, bool]:
        """
        Try to invoke the tool, return (output, success_flag).
        """
        name = call.function.name

        if name not in dispatch_map:
            return f"\nError: Unsupported tool call: {name}. Please check the tool name carefully.\n", False

        tool = dispatch_map[name]

        # Heavy initialize once
        if name not in heavy_init:
            try:
                tool.initialize()
            except Exception as e:
                return f"Initialization failed for {name}: {e}", False
            heavy_init.add(name)

        try:
            tool_args = json.loads(call.function.arguments)
        except json.JSONDecodeError:
            return f"Json decode error for tool call: {call.arguments}", False
        except Exception as e:
            return f"Unable to process tool call: {call.arguments}\n Error: {e}", False
        
        # Also input seq as inputs
        merged_args = {**tool_args, "session": session}
        try:
            result = tool(**merged_args)
            return result

        except TypeError as e:
            return f"Function call failed: {e}", False
        except Exception as e:
            return f"Tool runtime error in {name}: {e}", False

    def simple_process_messages(self, 
                        session: Session, 
                        out, 
                        config, 
                        logger, 
                        available_tool_dispatch,
                        available_tool_schemas,
                        heavy_init_cache
    ) -> None:

        # extract text from the content, no thinking for now
        text, thinking, tool_use = self.log_response(out, logger)

        if thinking:
            session.add_to_history("\nTHINKING BLOCK:\n" + thinking + "\n")
            session.output += "\nTHINKING BLOCK:\n" + thinking + "\n"

        session.add_to_history("\nANSWER:\n" + text)
        # Append generated text to prompt and output
        session.add_to_prompt(out.message.to_dict()) # Use to_dict() so that we can serialize
        session.output += "\nANSWER:\n" + text

        if tool_use:
            session.add_to_history("\nTOOL USE:\n" + tool_use + "\n")
            session.output += ("\nTOOL USE:\n" + tool_use + "\n")

        # means we need to invoke the tool
        if out.finish_reason == 'tool_calls':
            # check if the LLM calls the correct tool
            
            tool_calls = out.message.tool_calls
            # assert len(tool_calls) == 1, "Only one tool call is expected at a time as parallel_tool_use is False."
            
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                # TODO: There is a case where model has one problematic tool result and one successful tool result, 
                # In this case, we will have two messages, which is uncommon for Anthropic messages, should we fix that?
                output, success = self.dispatch_tool_call(
                    call           = tool_call,
                    dispatch_map   = available_tool_dispatch,
                    heavy_init     = heavy_init_cache,
                    session        = session,
                )
                if not success:
                    output = json.dumps({"Error": output})

                session.add_to_prompt({
                    "role":             "tool",
                    "tool_call_id":     tool_call.id,
                    "name":             tool_name,
                    "content":          output,
                })
                log_msg = f"\n<Tool result of {tool_name} starts here.>\n{output}<Tool result of {tool_name} ends here.>\n"
                if config.tool.append_tool_output:
                    session.output += log_msg
                session.add_to_history(log_msg)
                if logger:
                    logger.info(log_msg)
        elif out.finish_reason == 'stop':
            # If no search query needs to be executed, mark the sequence as finished
            session.finish()
            if logger:
                logger.info("Sequence marked as complete.")
        else:
            raise ValueError(
                f"Unexpected stop reason: {out.finish_reason}. Expected 'tool_calls' or 'stop'."
            )
        
    def process_messages(self, 
                        session: Session, 
                        out, 
                        config, 
                        logger, 
                        available_tool_dispatch,
                        available_tool_schemas,
                        heavy_init_cache,
                        is_multi_turn,
    ) -> bool:
        """
        Process the output from the model, dispatch tool calls, and update the sequence.
        Handles parallel tool calls.
        Returns:
            bool: True if a tool call was made, False otherwise.
        """
        message = out.message
        
        # Convert the message object to a JSON-serializable dictionary before appending
        message_dict = message.model_dump()
        session.add_to_prompt(message_dict)
        
        if message.tool_calls:
            logger.info(f"-> Planner proposed tool call(s): {message.tool_calls}")
            
            # Loop through ALL tool calls proposed by the model
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                
                tool_result, success = self.dispatch_tool_call(
                    tool_call,
                    session,
                    available_tool_dispatch,
                    heavy_init_cache,
                )
                
                # Append a tool response message for EACH tool call
                session.add_to_prompt({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": str(tool_result), # Ensure content is a string
                })

                if success:
                    logger.info(f"<- Tool {tool_name} returned: {str(tool_result)}")
                else:
                    logger.error(f"<- Tool {tool_name} failed: {tool_result}")

            return True # Signal that tool(s) were called
        else:
            # The model generated a regular text response
            text_response = message.content if message.content else ""
            logger.info(f"-> Planner proposed textual response: {text_response}")
            # The assistant's text response is already in the history.
            # Signal that NO tool was called.
            return False

class OpenAIResponsesClient(BaseRemoteAPILLM):
    def _init_client(self, config):
        # temp and top_p not supported
        if self.model_name == "gpt-o4-mini":
            self.model_name = "o4-mini"
            self.reasoning_effort = config.get("reasoning_effort", "low")
            self.reasoning_summary = config.get("reasoning_summary", "auto")
        elif self.model_name == "gpt-o3":
            self.model_name = "o3"
            self.reasoning_effort = config.get("reasoning_effort", "low")
            self.reasoning_summary = config.get("reasoning_summary", "auto")
        elif self.model_name == "gpt-5":
            # Keep model name as-is for Responses API, default to medium reasoning
            self.model_name = "gpt-5"
            self.reasoning_effort = config.get("reasoning_effort", "medium")
            self.reasoning_summary = config.get("reasoning_summary", "auto")
            # Text verbosity control for gpt-5 Responses API
            self.text_verbosity = config.get("text_verbosity", "medium")
        else:
            raise ValueError(f"Unseen supported model {self.model_name} for Response API")

        provider = _openai_provider_from_config(config)
        self.openai_provider = provider

        if provider == "azure":
            from openai import AzureOpenAI
            self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Azure OpenAI: set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and OPENAI_API_VERSION."
                )
            self.api_version = os.environ.get("OPENAI_API_VERSION")
            assert self.api_version == "2025-04-01-preview", (
                "Azure Responses API for reasoning models only supported with OPENAI_API_VERSION=2025-04-01-preview; "
                f"current: {self.api_version}"
            )
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            )
        else:
            from openai import OpenAI
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OpenAI API (api.openai.com): set OPENAI_API_KEY for Responses API, or use openai_provider=azure."
                )
            kwargs = {"api_key": self.api_key}
            base_url = os.environ.get("OPENAI_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
            self.client = OpenAI(**kwargs)

    def generate(self, prompts: List[Union[str, List[dict]]], sampling_params: Optional[SamplingParams] = None, tools: Optional[List[dict]] = None, **kwargs) -> List:
        if sampling_params is None:
            sampling_params = SamplingParams()
        responses = []
        raw_text = []
        for prompt in prompts:
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt     
            request_args = {
                "model": self.model_name,
                "input": messages,
                "max_output_tokens": sampling_params.max_tokens,
                "reasoning": {
                    "effort": self.reasoning_effort,
                    "summary": self.reasoning_summary,
                },
            }
            # Only gpt-5 supports text verbosity in Responses API
            if self.model_name == "gpt-5":
                verbosity = (self.text_verbosity or "medium").lower()
                if verbosity not in {"low", "medium", "high"}:
                    verbosity = "medium"
                request_args["text"] = {"verbosity": verbosity}
            # tool usage
            if tools is not None:
                request_args["tools"] = list(tools.values())
                # request_args["parallel_tool_calls"] = self.parallel_function_call
            response = self.client.responses.create(**request_args)

            # Note that response contains finish_reason, and messages
            responses.append(response.output)
            raw_text, _, _ = self.log_response(response.output)
        return responses, raw_text
    
    def invoke(self, prompt: str) -> type:
        """Method for code generation compatibility"""
        response = self.client.responses.create(
            model=self.model_name,
            input=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
        )
        raw_text, _, _ = self.log_response(response.output)
        return type("Response", (), {"content": raw_text})

    def log_response(self, out, logger=None):
        """Pretty print a message response with thinking blocks."""
        # logger.info("\n==== FULL RESPONSE ====")

        # deal with text
        text, thinking, tool_use = "", "", ""
        for block in out:
            if block.type == "reasoning":
                block_thinking = "\n".join([_summary.text for _summary in block.summary])
                if logger:
                    logger.info("\nTHINKING BLOCK:\n" + block_thinking)
                thinking += (block_thinking + "\n" if block_thinking else "")
            elif block.type == "function_call":
                if logger:
                    logger.info(f"\nTOOL USE: {block.name}\n" + block.arguments)
                tool_use += (block.name + "\n" + block.arguments)
            elif block.type == "message":
                assert len(block.content) == 1, f'length of content is not 1: {block.content}'
                if logger:
                    logger.info("\nANSWER:\n" + block.content[0].text)
                text += block.content[0].text
            else:
                raise ValueError(f"ERROR: Unseen block type {block.type}")
        
        # logger.info("\n==== END RESPONSE ====")
        
        return text, thinking, tool_use
    
    def dispatch_tool_call(
        self,
        call,
        session: Session,
        dispatch_map: Dict[str, BaseTool],
        heavy_init: Set[str],
    ) -> Tuple[str, bool]:
        """
        Try to invoke the tool, return (output, success_flag).
        """
        name = call.name

        if name not in dispatch_map:
            return f"Unsupported tool call: {name}.\n", False

        tool = dispatch_map[name]

        # Heavy initialize once
        if name not in heavy_init:
            try:
                tool.initialize()
            except Exception as e:
                return f"Initialization failed for {name}: {e}", False
            heavy_init.add(name)
        try:
            tool_args = json.loads(call.arguments)
        except json.JSONDecodeError:
            return f"Json decode error for tool call: {call.arguments}", False
        except Exception as e:
            return f"Unable to process tool call: {call.arguments}\n Error: {e}", False
            
        # Also input seq as inputs
        merged_args = {**tool_args, "session": session}

        try:
            result = tool(**merged_args)
            return result

        except TypeError as e:
            return f"Function call failed: {e}", False
        except Exception as e:
            return f"Tool runtime error in {name}: {e}", False

    def simple_process_messages(self, 
                        session: Session, 
                        out, 
                        config, 
                        logger, 
                        available_tool_dispatch,
                        available_tool_schemas,
                        heavy_init_cache) -> str:

        # extract text from the content, no thinking for now
        text, thinking, tool_use = self.log_response(out, logger)
        # means we need to invoke the tool
        
        if thinking:
            session.add_to_history("\nTHINKING BLOCK:\n" + thinking + "\n")
            session.output += "\nTHINKING BLOCK:\n" + thinking + "\n"

        session.add_to_history("\nANSWER:\n" + text)
        session.output += "\nANSWER:\n" + text

        # Append generated text to prompt and output
        for block in out:
            session.add_to_prompt(block.to_dict()) 

        if tool_use:
            session.add_to_history("\nTOOL USE:\n" + tool_use + "\n")
            session.output += ("\nTOOL USE:\n" + tool_use + "\n")

        # We do not have stop_reason for response API, so we need to track
        has_tool_calls = False
            
        for response in out:
            if response.type == "function_call":
                has_tool_calls = True

                tool_name = response.name
                # TODO: There is a case where model has one problematic tool result and one successful tool result, 
                # In this case, we will have two messages, which is uncommon for Anthropic messages, should we fix that?
                output, success = self.dispatch_tool_call(
                    call           = response,
                    dispatch_map   = available_tool_dispatch,
                    heavy_init     = heavy_init_cache,
                    session        = session,
                )
                # It seems that OpenAI wants us to use json format for error. Though it is optional.
                if not success:
                    output = json.dumps({"Error": output,})

                session.add_to_prompt({
                    "type":             "function_call_output",
                    "call_id":          response.call_id,
                    "output":           output,
                })
                log_msg = f"\n<Tool result of {tool_name} starts here.>\n{output}<Tool result of {tool_name} ends here.>\n"
                if config.tool.append_tool_output:
                    session.output += log_msg
                session.add_to_history(log_msg)
                if logger:
                    logger.info(log_msg)

        if not has_tool_calls:
            # If no tools are called, mark the sequence as finished
            session.finish()
            if logger:
                logger.info("Sequence marked as complete.")
        
    def process_messages(self, 
                        session: Session, 
                        out, 
                        config, 
                        logger, 
                        available_tool_dispatch,
                        available_tool_schemas,
                        heavy_init_cache,
                        is_multi_turn) -> bool:

        # extract text from the content, no thinking for now
        text, thinking, tool_use = self.log_response(out, logger)
        # means we need to invoke the tool
        
        if thinking:
            session.add_to_history("\nTHINKING BLOCK:\n" + thinking + "\n")
            session.output += "\nTHINKING BLOCK:\n" + thinking + "\n"

        session.add_to_history("\nANSWER:\n" + text)
        session.output += "\nANSWER:\n" + text

        # Append generated text to prompt and output
        for block in out:
            session.add_to_prompt(block.to_dict()) 

        if tool_use:
            session.add_to_history("\nTOOL USE:\n" + tool_use + "\n")
            session.output += ("\nTOOL USE:\n" + tool_use + "\n")
            
        if not is_multi_turn:
            session.finish()
            logger.info("Sequence marked as complete.")
            return

        # We do not have stop_reason for response API, so we need to track
        has_tool_calls = False

        for response in out:
            if response.type == "function_call":
                has_tool_calls = True

                tool_name = response.name
                # TODO: There is a case where model has one problematic tool result and one successful tool result, 
                # In this case, we will have two messages, which is uncommon for Anthropic messages, should we fix that?
                output, success = self.dispatch_tool_call(
                    call           = response,
                    dispatch_map   = available_tool_dispatch,
                    heavy_init     = heavy_init_cache,
                    session        = session,
                )
                # It seems that OpenAI wants us to use json format for error. Though it is optional.
                if not success:
                    output = json.dumps({"Error": output,})

                session.add_to_prompt({
                    "type":             "function_call_output",
                    "call_id":          response.call_id,
                    "output":           output,
                })
                log_msg = f"\n<Tool result of {tool_name} starts here.>\n{output}<Tool result of {tool_name} ends here.>\n"
                if config.tool.append_tool_output:
                    session.output += log_msg
                session.add_to_history(log_msg)
                logger.info(log_msg)

        if not has_tool_calls:
            return False
        else:
            return True
        
class AnthropicClient(BaseRemoteAPILLM):
    def _init_client(self, config):
        use_aws = os.environ.get("USE_ANTHROPIC_AWS", "false").lower() == "true"
        if use_aws:
            from anthropic import AnthropicBedrock
            aws_access_key = os.environ.get("ANTHROPIC_AWS_ACCESS_KEY")
            aws_secret_key = os.environ.get("ANTHROPIC_AWS_SECRET_KEY")
            if not aws_access_key or not aws_secret_key:
                raise ValueError("Anthropic AWS credentials required. Set via environment variables 'ANTHROPIC_AWS_ACCESS_KEY' and 'ANTHROPIC_AWS_SECRET_KEY'.")
            aws_region = os.environ.get("ANTHROPIC_AWS_REGION", "us-west-2")
            self.anthropic = AnthropicBedrock(
                aws_access_key=aws_access_key,
                aws_secret_key=aws_secret_key,
                aws_region=aws_region,
            )
        else: # Use Anthropic's API directly
            from anthropic import Anthropic
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("Anthropic API key required. Set via environment variable 'ANTHROPIC_API_KEY'.")
            self.anthropic = Anthropic(api_key=self.api_key)
        self.provider = 'anthropic'

    def generate(self, prompts: List[Union[str, List[dict]]],  sampling_params: Optional[SamplingParams] = None, tools: Optional[List[dict]] = None, **kwargs) -> List:
        if sampling_params is None:
            sampling_params = SamplingParams()
        responses = []
        raw_text = []
        for prompt in prompts:
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            assert isinstance(prompt, list), "For Anthropic, prompts should be a list of dictionaries with 'role' and 'content' keys. But we got {}".format(prompt)
            request_args = {
                "model": self.model_name,
                "max_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "top_k": sampling_params.top_k,
                "stop_sequences": sampling_params.stop,
                "messages": prompt,
            }
            # extended thinking
            if kwargs.get("max_think_tokens", 0) > 0:
                max_think_tokens = kwargs.get("max_think_tokens")
                request_args["thinking"] = {"type": "enabled", "budget_tokens": max_think_tokens}
                request_args["temperature"] = 0.001
                del request_args["top_p"]
            if tools is not None:
                request_args["tools"] = list(tools.values())
                request_args["tool_choice"] = {
                    "type": "auto",
                }
            response = self.anthropic.messages.create(**request_args)
            responses.append(response)

            text, _, _ = self.log_response(response.content)

            raw_text.append(text)
        return responses, raw_text

    def log_response(self, content, logger=None):
        """Pretty print a message response with thinking blocks."""
        # logger.info("\n==== FULL RESPONSE ====")
        text, thinking, tool_use = "", "", ""
        for block in content:
            if block.type == "thinking":
                if logger:
                    logger.info("\n🧠 THINKING BLOCK:\n" + block.thinking)
                thinking += block.thinking
                # logger.info(f"\n[Signature available: {bool(getattr(block, 'signature', None))}]")
                # if hasattr(block, 'signature') and block.signature:
                #     logger.info(f"[Signature (first 50 chars): {block.signature[:50]}...]")
            elif block.type == "redacted_thinking":
                if logger:
                    logger.info(
                        "\n🔒 REDACTED THINKING BLOCK:\n"
                        + f"[Data length: {len(block.data) if hasattr(block, 'data') else 'N/A'}]"
                        )
            elif block.type == "tool_use":
                if logger:
                    logger.info(f"\n🔧 TOOL USE: {block.name}\n" + str(block.input))
                tool_use += (block.name + "\n" + str(block.input))
            elif block.type == "text":
                if logger:
                    logger.info("\n✅ ANSWER:\n" + block.text)
                text += block.text
            else:
                raise ValueError(f"ERROR: Unseen block type {block.type}")
        
        # logger.info("\n==== END RESPONSE ====")
        
        return text, thinking, tool_use

    def dispatch_tool_call(
        self,
        call,
        session: Session,
        dispatch_map: Dict[str, BaseTool],
        heavy_init: Set[str],
    ) -> Tuple[str, bool]:
        """
        Try to invoke the tool, return (output, success_flag).
        """
        name = call.name

        if name not in dispatch_map:
            return f"\nError: Unsupported tool call: {name}.\n", False

        tool = dispatch_map[name]

        # Heavy initialize once
        if name not in heavy_init:
            try:
                tool.initialize()
            except Exception as e:
                return f"Initialization failed for {name}: {e}", False
            heavy_init.add(name)

        # Also input seq as inputs
        merged_args = {**call.input, "session": session}

        try:
            result = tool(**merged_args)
            return result

        except TypeError as e:
            return f"Function call failed: {e}", False
        except Exception as e:
            return f"Tool runtime error in {name}: {e}", False

    def simple_process_messages(self, 
                        session: Session, 
                        out, 
                        config, 
                        logger, 
                        available_tool_dispatch,
                        available_tool_schemas,
                        heavy_init_cache) -> str:

        # extract text from the content
        text, thinking, tool_use = self.log_response(out.content, logger)

        if thinking:
            session.add_to_history("\n🧠 THINKING BLOCK:\n" + thinking + "\n")
            session.output += "\n🧠 THINKING BLOCK:\n" + thinking + "\n"
            
        session.add_to_history("\n✅ ANSWER:\n" + text)
        session.output += "\n✅ ANSWER:\n" + text

        # Append generated text to prompt and output
        session.add_to_prompt({
            "role": "assistant",
            "content": [block.to_dict() for block in out.content], # Use to_dict() for json saving
        })
        
        if tool_use:
            session.add_to_history("\n🔧 TOOL USE:\n" + tool_use + "\n")
            session.output += ("\n🔧 TOOL USE:\n" + tool_use + "\n")

        # means we need to invoke the tool
        if out.stop_reason == "tool_use":
            content_to_return = []
            # identify the message related to tool_use
            for tool_call in out.content:
                if tool_call.type == "tool_use":
                    # TODO: There is a case where model has one problematic tool result and one successful tool result, 
                    # In this case, we will have two messages, which is uncommon for Anthropic messages, should we fix that?
                    output, success = self.dispatch_tool_call(
                        call           = tool_call,
                        dispatch_map   = available_tool_dispatch,
                        heavy_init     = heavy_init_cache,
                        session        = session,
                    )
                    content_to_return.append({
                        "type":         "tool_result",
                        "tool_use_id":  tool_call.id,
                        "content":      output,
                        "is_error":    not success,
                    })
                    log_msg = f"\n<Tool result of {tool_call.name} starts here.>\n{output}<Tool result of {tool_call.name} ends here.>\n"
                    if config.tool.append_tool_output:
                        session.output += log_msg
                    session.add_to_history(log_msg)
                    if logger:
                        logger.info(log_msg)

            message = {
                "role":         "user",
                "content":      content_to_return,
            }
            session.add_to_prompt(message)

        elif out.stop_reason == "end_turn":
            # If no search query needs to be executed, mark the sequence as finished
            session.finish()
            if logger:
                logger.info("Sequence marked as complete.")
        else:
            raise ValueError(
                f"Unexpected stop reason: {out.stop_reason}. Expected 'tool_use' or 'stop_sequence'."
            )

    def process_messages(self, 
                        session: Session, 
                        out, 
                        config, 
                        logger, 
                        available_tool_dispatch,
                        available_tool_schemas,
                        heavy_init_cache,
                        is_multi_turn) -> str:

        # extract text from the content
        text, thinking, tool_use = self.log_response(out.content, logger)

        if thinking:
            session.add_to_history("\n🧠 THINKING BLOCK:\n" + thinking + "\n")
            session.output += "\n🧠 THINKING BLOCK:\n" + thinking + "\n"
            
        session.add_to_history("\n✅ ANSWER:\n" + text)
        session.output += "\n✅ ANSWER:\n" + text

        # Append generated text to prompt and output
        session.add_to_prompt({
            "role": "assistant",
            "content": [block.to_dict() for block in out.content], # Use to_dict() for json saving
        })
        
        if tool_use:
            session.add_to_history("\n🔧 TOOL USE:\n" + tool_use + "\n")
            session.output += ("\n🔧 TOOL USE:\n" + tool_use + "\n")

        # means we need to invoke the tool (if multi turn)
        if out.stop_reason == "end_turn" or not is_multi_turn:
            # If no search query needs to be executed, mark the sequence as finished
            session.finish()
            logger.info("Sequence marked as complete.")
            return False
        elif out.stop_reason == "tool_use":
            content_to_return = []
            # identify the message related to tool_use
            for tool_call in out.content:
                if tool_call.type == "tool_use":
                    # TODO: There is a case where model has one problematic tool result and one successful tool result, 
                    # In this case, we will have two messages, which is uncommon for Anthropic messages, should we fix that?
                    output, success = self.dispatch_tool_call(
                        call           = tool_call,
                        dispatch_map   = available_tool_dispatch,
                        heavy_init     = heavy_init_cache,
                        session        = session,
                    )
                    content_to_return.append({
                        "type":         "tool_result",
                        "tool_use_id":  tool_call.id,
                        "content":      output,
                        "is_error":    not success,
                    })
                    log_msg = f"\n<Tool result of {tool_call.name} starts here.>\n{output}<Tool result of {tool_call.name} ends here.>\n"
                    if config.tool.append_tool_output:
                        session.output += log_msg
                    session.add_to_history(log_msg)
                    logger.info(log_msg)

            message = {
                "role":         "user",
                "content":      content_to_return,
            }
            session.add_to_prompt(message)
            return True

    def invoke(self, prompt: str) -> type:
        """Method for code generation compatibility"""
        response = self.anthropic.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.7,
        )
        return type("Response", (), {"content": response.content[0].text}) 

class QWENClient(OpenAIChatCompletionClient):
    def _init_client(self, config):
        from openai import OpenAI
        port = config.get("port", 8000)
        if not port:
            port = 8000
        self.client = OpenAI(
            api_key="I*am*an*API*Key",
            base_url=f"http://localhost:{port}/v1",
        )
        self.provider = 'qwen'

class GPTOSSClient(BaseRemoteAPILLM):
    def _init_client(self, config):
        # Normalize model name - map gpt-oss-20b to the full model name expected by the server
        if self.model_name == "gpt-oss-20b":
            self.model_name = "openai/gpt-oss-20b"
        
        # Configure reasoning effort/summary from config
        self.reasoning_effort = config.get("reasoning_effort", "low")
        self.reasoning_summary = config.get("reasoning_summary", "auto")

        # Connect to a locally hosted OpenAI-compatible server via port
        from openai import OpenAI as _VLLMOpenAI
        port = config.get("port", 8000)
        self.client = _VLLMOpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{port}/v1",
        )
        self.provider = 'gpt-oss'

    def generate(self, prompts: List[Union[str, List[dict]]], sampling_params: Optional[SamplingParams] = None, tools: Optional[List[dict]] = None, **kwargs) -> List:
        """Custom generate method for GPTOSSClient using responses API with temperature and top_p support."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        responses = []
        raw_text = []
        
        for prompt in prompts:
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt     
            
            request_args = {
                "model": self.model_name,
                "input": messages,  # responses API uses 'input' not 'messages'
                "max_output_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "reasoning": {
                    "effort": self.reasoning_effort,
                    "summary": self.reasoning_summary,
                },
            }
            
            # Add stop sequences if specified
            if sampling_params.stop:
                request_args["stop"] = sampling_params.stop
            
            # Add tool usage if specified
            if tools is not None:
                request_args["tools"] = list(tools.values())
                request_args["tool_choice"] = "auto"
            
            # Remove None values to avoid API errors
            request_args = {k: v for k, v in request_args.items() if v is not None}
            
            # Use responses.create for OSS model (similar to test file pattern)
            response = self.client.responses.create(**request_args)
            
            # Note that response contains finish_reason, and messages
            responses.append(response.output)
            raw_text, _, _ = self.log_response(response.output)
            
        return responses, raw_text

    def invoke(self, prompt: str) -> type:
        """Custom invoke method for GPTOSSClient using responses API."""
        response = self.client.responses.create(
            model=self.model_name,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=2000,
            temperature=0.001,
            reasoning={
                "effort": self.reasoning_effort,
                "summary": self.reasoning_summary,
            },
        )
        raw_text, _, _ = self.log_response(response.output)
        return type("Response", (), {"content": raw_text})

    def log_response(self, out, logger=None):
        """Custom log_response method for GPTOSSClient to handle OSS model response format."""
        text, thinking, tool_use = "", "", ""
        
        # The OSS model returns an array of message objects
        if isinstance(out, list):
            messages = out
        else:
            # Handle case where out might be a single object
            messages = [out] if hasattr(out, 'type') else []

        for message in messages:
            if hasattr(message, 'type'):
                if message.type == "reasoning":
                    # Extract thinking content from reasoning blocks
                    if hasattr(message, 'content') and message.content:
                        reasoning_text = ""
                        for content_item in message.content:
                            if hasattr(content_item, 'text'):
                                reasoning_text += content_item.text + "\n"
                            elif isinstance(content_item, dict) and 'text' in content_item:
                                reasoning_text += content_item['text'] + "\n"
                        
                        if reasoning_text.strip():
                            if logger:
                                logger.info("\nTHINKING BLOCK:\n" + reasoning_text)
                            thinking += reasoning_text

                elif message.type == "function_call":
                    # Handle function calls
                    tool_name = message.name if hasattr(message, 'name') else "unknown"
                    arguments = message.arguments if hasattr(message, 'arguments') else "{}"
                    if logger:
                        logger.info(f"\nTOOL USE: {tool_name}\n" + arguments)
                    tool_use += (tool_name + "\n" + arguments)

                elif message.type == "message" or not hasattr(message, 'type'):
                    # Handle regular message content
                    if hasattr(message, 'content'):
                        if isinstance(message.content, list):
                            for content_item in message.content:
                                if hasattr(content_item, 'text'):
                                    text += content_item.text
                                elif isinstance(content_item, dict) and 'text' in content_item:
                                    text += content_item['text']
                        elif isinstance(message.content, str):
                            text += message.content
                    
                    if text.strip() and logger:
                        logger.info("\nANSWER:\n" + text)

        return text, thinking, tool_use

    def simple_process_messages(self, 
                        session: Session, 
                        out, 
                        config, 
                        logger, 
                        available_tool_dispatch,
                        available_tool_schemas,
                        heavy_init_cache) -> None:
        """Custom simple_process_messages method for GPTOSSClient to handle OSS model response format."""
        
        # Extract text, thinking, and tool use
        text, thinking, tool_use = self.log_response(out, logger)
        
        if thinking:
            session.add_to_history("\nTHINKING BLOCK:\n" + thinking + "\n")
            session.output += "\nTHINKING BLOCK:\n" + thinking + "\n"

        if text:
            session.add_to_history("\nANSWER:\n" + text)
            session.output += "\nANSWER:\n" + text

        if tool_use:
            session.add_to_history("\nTOOL USE:\n" + tool_use + "\n")
            session.output += "\nTOOL USE:\n" + tool_use + "\n"

        # Process each message in the response
        messages = out if isinstance(out, list) else [out] if hasattr(out, 'type') else []
        has_tool_calls = False

        for message in messages:
            if hasattr(message, 'type'):
                # Add message to prompt
                session.add_to_prompt(message.to_dict() if hasattr(message, 'to_dict') else message.__dict__)
                
                if message.type == "function_call":
                    has_tool_calls = True
                    tool_name = message.name if hasattr(message, 'name') else "unknown"
                    
                    # Dispatch the tool call
                    output, success = self.dispatch_tool_call(
                        call=message,
                        dispatch_map=available_tool_dispatch,
                        heavy_init=heavy_init_cache,
                        session=session,
                    )
                    
                    if not success:
                        output = json.dumps({"Error": output})
                    
                    # Add tool result to prompt
                    session.add_to_prompt({
                        "type": "function_call_output",
                        "call_id": message.call_id if hasattr(message, 'call_id') else getattr(message, 'id', 'unknown'),
                        "output": output,
                    })
                    
                    # Log tool result
                    log_msg = f"\n<Tool result of {tool_name} starts here.>\n{output}<Tool result of {tool_name} ends here.>\n"
                    if config.tool.append_tool_output:
                        session.output += log_msg
                    session.add_to_history(log_msg)
                    if logger:
                        logger.info(log_msg)

        # If no tool calls, mark sequence as finished
        if not has_tool_calls:
            session.finish()
            if logger:
                logger.info("Sequence marked as complete.")

    def process_messages(self, 
                        session: Session, 
                        out, 
                        config, 
                        logger, 
                        available_tool_dispatch,
                        available_tool_schemas,
                        heavy_init_cache,
                        is_multi_turn) -> bool:
        """Custom process_messages method for GPTOSSClient to handle OSS model response format."""
        
        # Extract text, thinking, and tool use
        text, thinking, tool_use = self.log_response(out, logger)
        
        if thinking:
            session.add_to_history("\nTHINKING BLOCK:\n" + thinking + "\n")
            session.output += "\nTHINKING BLOCK:\n" + thinking + "\n"

        if text:
            session.add_to_history("\nANSWER:\n" + text)
            session.output += "\nANSWER:\n" + text

        if tool_use:
            session.add_to_history("\nTOOL USE:\n" + tool_use + "\n")
            session.output += "\nTOOL USE:\n" + tool_use + "\n"

        # Process each message in the response
        messages = out if isinstance(out, list) else [out] if hasattr(out, 'type') else []
        has_tool_calls = False

        for message in messages:
            if hasattr(message, 'type'):
                # Add message to prompt
                session.add_to_prompt(message.to_dict() if hasattr(message, 'to_dict') else message.__dict__)
                
                if message.type == "function_call":
                    has_tool_calls = True
                    tool_name = message.name if hasattr(message, 'name') else "unknown"
                    
                    # Dispatch the tool call
                    output, success = self.dispatch_tool_call(
                        call=message,
                        dispatch_map=available_tool_dispatch,
                        heavy_init=heavy_init_cache,
                        session=session,
                    )
                    
                    if not success:
                        output = json.dumps({"Error": output})
                    
                    # Add tool result to prompt
                    session.add_to_prompt({
                        "type": "function_call_output",
                        "call_id": message.call_id if hasattr(message, 'call_id') else getattr(message, 'id', 'unknown'),
                        "output": output,
                    })
                    
                    # Log tool result
                    log_msg = f"\n<Tool result of {tool_name} starts here.>\n{output}<Tool result of {tool_name} ends here.>\n"
                    if config.tool.append_tool_output:
                        session.output += log_msg
                    session.add_to_history(log_msg)
                    if logger:
                        logger.info(log_msg)

        # Handle session completion logic
        if not has_tool_calls and not is_multi_turn:
            session.finish()
            if logger:
                logger.info("Sequence marked as complete.")
            return False
        
        return has_tool_calls

    def dispatch_tool_call(
        self,
        call,
        session: Session,
        dispatch_map: Dict[str, BaseTool],
        heavy_init: Set[str],
    ) -> Tuple[str, bool]:
        """Custom dispatch_tool_call method for GPTOSSClient to handle OSS model tool call format."""
        name = call.name if hasattr(call, 'name') else "unknown"

        if name not in dispatch_map:
            return f"Unsupported tool call: {name}.\n", False

        tool = dispatch_map[name]

        # Heavy initialize once
        if name not in heavy_init:
            try:
                tool.initialize()
            except Exception as e:
                return f"Initialization failed for {name}: {e}", False
            heavy_init.add(name)

        # Parse tool arguments
        try:
            if hasattr(call, 'arguments'):
                if isinstance(call.arguments, str):
                    tool_args = json.loads(call.arguments)
                else:
                    tool_args = call.arguments
            else:
                tool_args = {}
        except json.JSONDecodeError as e:
            return f"Failed to parse tool arguments: {e}", False

        # Also input session as input
        merged_args = {**tool_args, "session": session}

        try:
            result = tool(**merged_args)
            return result

        except TypeError as e:
            return f"Function call failed: {e}", False
        except Exception as e:
            return f"Tool runtime error in {name}: {e}", False

def setup_dspy_model(model_name: str):
    """
    Configure and return a DSPy language model (optional; requires ``pip install dspy``).
    Not used by the default datasci agent path.
    """
    import dspy

    model_configs = {
        'o1': {
            'model_path': 'openai/o1',
            'kwargs': {
                'api_key': os.getenv("OPENAI_API_KEY"),
                'temperature': 0.001,
                'top_p': 0.9,
                'max_tokens': 8192
            }
        },
        'gpt-4o': {
            'model_path': 'openai/gpt-4o',
            'kwargs': {
                'api_key': os.getenv("OPENAI_API_KEY"),
                'temperature': 0.001,
                'top_p': 0.9
            }
        },
        'deepseek': {
            'model_path': 'openai/deepseek-reasoner',
            'kwargs': {
                'api_key': os.getenv("DEEPSEEK_API_KEY"),
                'temperature': 0.001,
                'top_p': 0.9
            }
        },
        'deepseek-v3': {
            'model_path': 'openai/DeepSeek-V3',
            'kwargs': {
                'api_key': os.getenv("DEEPSEEK_API_KEY"),
                'temperature': 0.001,
                'top_p': 0.9
            }
        }
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unsupported model: {model_name}")
    
    config = model_configs[model_name]
    return dspy.LM(config['model_path'], **config['kwargs'])