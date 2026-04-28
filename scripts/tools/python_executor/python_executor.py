from ..base import BaseTool, register_tool
import os
import time
import base64
import requests
from requests.exceptions import ConnectionError, Timeout
import random
import json

def backoff_delay(attempt, base=1.0, factor=2.0, cap=30.0):
    """Compute an exponential backoff with full jitter."""
    # exponential part
    delay = min(cap, base * (factor ** attempt))
    # jitter it down to between 0.5× and 1× of that delay
    return random.uniform(delay * 0.5, delay)

def with_kwargs(d:dict, **kwargs):
    return {**d, **kwargs}

def validate_inputs(func):
    def wrapper(self, code, files_to_load, files_to_save, **kwargs):
        if not isinstance(files_to_load, list):
            raise TypeError("files_to_load must be a list")
        if not isinstance(files_to_save, list):
            raise TypeError("files_to_save must be a list")
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        return func(self, code, files_to_load, files_to_save, **kwargs)
    return wrapper

@register_tool
class python_executor(BaseTool):
    name = "python_executor"
    # tool_description = (
    #     "Run a Python script in an isolated HTTP sandbox with a {timeout}-second limit. Execution is single-shot and non-interactive (no REPL, no persistent state). "
    #     "Before running, upload any `files_to_load`; after completion, download any `files_to_save`. The maximum file size limit is {max_file_size_mb} MB. The tool returns the complete program output, including both stdout and stderr streams. "
    #     "Use explicit `print(...)` calls to emit values you want captured in the output. You can call this tool up to {max_turns} times in a conversation."
    # )
    tool_description = (
        "Execute a Python script in an isolated HTTP sandbox with a {timeout}-second time limit. "
        "Each run is single-shot and stateless (no REPL, no persistent environment between runs). "
        "You may upload input files via `files_to_load` and retrieve results via `files_to_save`. "
        "The maximum file size for both upload and download is {max_file_size_mb} MB. "
        "The tool returns the full program output, including both stdout and stderr. "
        "Use explicit `print(...)` statements to capture values in the output. "
        "This tool can be invoked up to {max_turns} times per conversation."
    )
    tool_parameters = {
      "type": "object",
      "properties": {
        "code": {
          "type": "string",
          "description": "The Python code to execute."
        },
        "files_to_load": {
          "type": "array",
          "items": { "type": "string" },
          "description": "List of input file paths to upload prior to execution (e.g. [\"input1.csv\", \"config.json\"])."
        },
        "files_to_save": {
          "type": "array",
          "items": { "type": "string" },
          "description": "List of output file paths to download after execution (e.g. [\"results.csv\", \"log.txt\"])."
        }
      },
      "required": ["code", "files_to_load", "files_to_save"]
    }

    def initialize(self, config):
        self.url = config.tool.python_executor.get("url", "http://code-executor:8080/run_code")
        self.num_retries = config.tool.python_executor.get("num_retries", 3)
        self.timeout = config.tool.python_executor.get("timeout", 100)

        # the work_dir is the directory where the files are loaded from and saved to
        self.work_dir = config.cache_dir
        print("work_dir set to", self.work_dir)
        self.max_file_size = config.tool.python_executor.get("max_file_size_mb", 50) * 1024 * 1024
    
    @validate_inputs
    def __call__(
        self,
        code: str,
        files_to_load: dict,
        files_to_save: dict,
        **kwargs,
    ):
        encoded_files = {}
        for fname in files_to_load:
            if os.path.basename(fname) != fname:
                raise ValueError(f"Invalid file path: {fname!r}. Must be a direct child of work_dir.")
            full_path = os.path.join(self.work_dir, fname)
            if not os.path.isfile(full_path):
                raise FileNotFoundError(f"File not found: {fname}")
            
            # Check file size before reading
            file_size = os.path.getsize(full_path)
            if file_size > self.max_file_size:
                max_size_mb = self.max_file_size / (1024 * 1024)
                actual_size_mb = file_size / (1024 * 1024)
                raise ValueError(f"File {fname} ({actual_size_mb:.2f}MB) exceeds maximum size limit of {max_size_mb:.2f}MB")
            
            with open(full_path, 'rb') as f:
                b = f.read()
            encoded_files[fname] = base64.b64encode(b).decode('utf-8')

        # 2. Prepare payload
        payload = {
            'code': code,
            'language': 'python',
            'files': encoded_files,
            'fetch_files': files_to_save,
            "compile_timeout": self.timeout,
            "run_timeout": self.timeout,
        }

# 3. Retry loop
        attempts = 0
        while True:
            try:
                resp = requests.post(self.url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                break
            except ConnectionError as e:
                attempts += 1
                if self.num_retries >= 0 and attempts > self.num_retries:
                    return {"stdout": "", "stderr": f"Failed after {attempts} attempts: {type(e).__name__}: {e}"}, False
                # otherwise sleep then retry
                time.sleep(backoff_delay(attempts))
                continue
            except requests.exceptions.HTTPError as e:
                return {"stdout": "", "stderr": f"HTTP Error: {e.response.status_code} - {e.response.text}"}, False
            except Exception as e:
                return {"stdout": "", "stderr": f"Error: {type(e).__name__}: {e}"}, False
                
        run_result = data["run_result"]
        run_result.pop("execution_time") # remove execution_time for greedy decoding consistency
        # run_result = {k: run_result[k] for k in ["stdout", "stderr",]}
        if data["files"]:
            failed_files = []
            for file_name, b64 in data["files"].items():
                if os.path.basename(file_name) != file_name:
                    raise ValueError(f"Invalid file path: {file_name!r}. Must be a direct child of work_dir.")
                full_path = os.path.join(self.work_dir, file_name)
                try:
                    with open(full_path, 'wb') as f:
                        f.write(base64.b64decode(b64))
                        run_result["stdout"] += f"\nFile {file_name} saved successfully.\n"
                except Exception as e:
                    raise ValueError(f"Failed to save file {file_name}: {e}")
                    failed_files.append(f"{file_name}: {str(e)}")

            if failed_files:
                run_result["stderr"] += f"\nWarning: Failed to save files: {'; '.join(failed_files)}"
        return json.dumps(run_result), True

    def get_tool_description(self, model: str, config: dict):
        # TODO: this is a hack to get the timeout from the config
        timeout = config.tool.python_executor.get("timeout", 100)
        max_turns = config.get("max_turn", 10)
        max_file_size_mb = config.tool.python_executor.get("max_file_size_mb", 50)
        payload = {
            "timeout": timeout,
            "max_turns": max_turns,
            "max_file_size_mb": max_file_size_mb,
        }
        if "claude" in model.lower():
            return {
                "name": self.__class__.name,
                "description": self.__class__.tool_description.format(**payload),
                "input_schema": self.__class__.tool_parameters,
            }
        elif (model in ["gpt-o4-mini", "gpt-o3", "gpt-5"]) or ("gpt-oss" in model.lower()):
            return {
                "type": "function",
                "name": self.__class__.name,
                "description": self.__class__.tool_description.format(**payload),
                "parameters": with_kwargs(self.__class__.tool_parameters, additionalProperties=False),
                "strict": True,
            }
        elif  ('gpt' in model.lower()) or ('qwen' in model.lower()):
            return {
                "type": "function",
                "function": {
                    "name": self.__class__.name,
                    "description": self.__class__.tool_description.format(**payload),
                    "parameters": with_kwargs(self.__class__.tool_parameters, additionalProperties=False),
                    "strict": True,
                }
            }
        else:
            raise ValueError(f"Unsupported model: {model}. Supported models are: claude, gpt.")
