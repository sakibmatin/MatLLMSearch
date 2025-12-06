import asyncio
import threading
from typing import Any, Dict, List, Optional, Union
import yaml
from copy import deepcopy
import gc
import os
import re


import weave
WEAVE_AVAILABLE = True

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import litellm
from vllm import LLM, SamplingParams
VLLM_AVAILABLE = True

def load_models_and_credentials(models_file="models.yaml", credentials_file="credentials.yaml"):
    """Load configuration files with proper error handling."""
    try:
        with open(models_file, "r") as f:
            models = yaml.safe_load(f)
        if models is None:
            raise ValueError(f"Models file {models_file} is empty or invalid")
    except FileNotFoundError:
        raise FileNotFoundError(f"Models configuration file not found: {models_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in models file {models_file}: {e}")
    
    try:
        with open(credentials_file, "r") as f:
            credentials = yaml.safe_load(f)
        if credentials is None:
            credentials = {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Credentials configuration file not found: {credentials_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in credentials file {credentials_file}: {e}")
    
    return models, credentials


def expand_env_vars_in_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Expand environment variables in credential dictionary values."""
    expanded = {}
    for key, value in d.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1]
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(f"Environment variable {var_name} is not set. Please set it or update credentials.yaml with the actual API key.")
            expanded[key] = env_value
        elif isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
            var_name = value[1:]
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(f"Environment variable {var_name} is not set. Please set it or update credentials.yaml with the actual API key.")
            expanded[key] = env_value
        else:
            expanded[key] = value
    return expanded


def load_model_config(model_name, models, credentials):
    try:
        model_config = deepcopy(models[model_name])
    except KeyError:
        raise KeyError(f"Model `{model_name}` not found in models_file")
    
    credentials_config = {}
    if 'credentials' in model_config:
        credentials_tag = model_config["credentials"]
        if credentials_tag is not None:
            try:
                credentials_config = credentials[credentials_tag]
                # Expand environment variables in credentials
                credentials_config = expand_env_vars_in_dict(credentials_config)
            except KeyError:
                raise KeyError(f"Credentials `{credentials_tag}` not found in credentials_file")
    
    model_config["model_name"] = model_name
    model_config["credentials"] = credentials_config

    return model_config


def validate_device(device):
    """
    Validate and return the appropriate device configuration.
    
    Args:
        device: Device specification (str or int or None)
        
    Returns:
        str: Valid device string
    """
    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    elif isinstance(device, int):
        if torch.cuda.is_available() and device < torch.cuda.device_count():
            return f"cuda:{device}"
        else:
            return "cpu"
    elif isinstance(device, str):
        if device.startswith("cuda"):
            if torch.cuda.is_available():
                return device
            else:
                print(f"Warning: CUDA requested but not available, falling back to CPU")
                return "cpu"
        return device
    else:
        raise ValueError(f"Invalid device specification: {device}")


class Generation:
    """
    A unified generator supporting OpenAI (GPT-4o, GPT-4, etc.), Google Gemini, 
    Anthropic Claude, and Hugging Face models with optional concurrency.
    """
    
    def __init__(
        self,
        models_file: str = "models.yaml",
        credentials_file: str = "credentials.yaml",
        model_name: Optional[str] = None,
        device: Optional[Union[str, int]] = None,
        max_workers: int = 4,
    ):
        """
        A unified generator supporting OpenAI (GPT-4o, GPT-4, etc.), Google Gemini, 
        Anthropic Claude, and Hugging Face models with optional concurrency.

        Args:
            models_file: Path to the models configuration file.
            credentials_file: Path to the credentials configuration file.
            device: Device identifier for HF models (e.g., "cuda" or 0). If None, will auto-detect.
            max_workers: Number of threads for concurrent generation.
        """

        self.models, self.credentials = load_models_and_credentials(models_file, credentials_file)

        self.model_name = model_name
        self.device = validate_device(device)

        self.hf_model_name = None
        self.hf_model = None
        self.hf_tokenizer = None
        self.generator = None
        self._model_lock = threading.Lock()

        self.vllm_model_name = None
        self.vllm_model = None

        self.executor = weave.ThreadPoolExecutor(max_workers=max_workers)
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Clean up resources."""
        if not self._closed:
            self.executor.shutdown(wait=True)
            self._cleanup_hf_model()
            self._cleanup_vllm_model()
            self._closed = True

    def __del__(self):
        if hasattr(self, '_closed') and not self._closed:
            self.close()

    def _cleanup_hf_model(self):
        """Clean up HuggingFace model resources."""
        if self.hf_model is not None:
            del self.hf_model
            self.hf_model = None
        if self.hf_tokenizer is not None:
            del self.hf_tokenizer
            self.hf_tokenizer = None
        if self.generator is not None:
            del self.generator
            self.generator = None
        self.hf_model_name = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _cleanup_vllm_model(self):
        """Clean up vLLM model resources."""
        if self.vllm_model is not None:
            del self.vllm_model
            self.vllm_model = None
        self.vllm_model_name = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _load_hf_model_and_tokenizer(self, model_name):
        """Load HuggingFace model with proper error handling and cleanup."""
        with self._model_lock:
            if self.hf_model_name == model_name and self.hf_model is not None:
                return
            
            self._cleanup_hf_model()
            
            try:
                self.hf_model_name = model_name
                self.hf_model = AutoModelForCausalLM.from_pretrained(model_name)
                self.hf_model.to(self.device)
                self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                if self.hf_tokenizer.pad_token is None:
                    self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
                
                self.generator = pipeline(
                    "text-generation",
                    model=self.hf_model,
                    tokenizer=self.hf_tokenizer,
                    device=0 if isinstance(self.device, int) or self.device == "cuda" else -1,
                )
            except Exception as e:
                self._cleanup_hf_model()
                raise RuntimeError(f"Failed to load HuggingFace model {model_name}: {e}")

    @weave.op()
    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text from the given prompt using the specified model.

        Args:
            prompt: Input prompt string.
            messages: List of message dictionaries for chat models.
            model_name: Model identifier that you set in models.yaml.
            **kwargs: Additional model-specific arguments.

        Returns:
            A dict containing the generated text and metadata.
        """
        if self._closed:
            raise RuntimeError("Generator has been closed")

        if model_name is None:
            model_name = self.model_name
        if model_name is None:
            raise ValueError("model_name is required")
        
        if prompt is None and messages is None:
            raise ValueError("Either prompt or messages must be provided")
        
        if prompt is not None and messages is not None:
            raise ValueError("Cannot provide both prompt and messages")
        
        model_config = load_model_config(model_name, self.models, self.credentials)

        if model_config["provider"] == "local":
            if messages is not None:
                raise NotImplementedError("Local models only support prompt generation currently.")
            if not prompt or not prompt.strip():
                raise ValueError("Prompt cannot be empty for local models")

            use_vllm = model_config.get("use_vllm", True)

            if use_vllm:
                return self._generate_vllm(
                    model_config=model_config,
                    prompt=prompt,
                    **kwargs,
                )
            else:
                return self._generate_hf(
                    model_config=model_config,
                    prompt=prompt,
                    **kwargs,
                )
        
        return self._generate_litellm(
            model_config=model_config,
            prompt=prompt,
            messages=messages,
            **kwargs,
        )
    
    @weave.op()
    def _generate_litellm(
        self,
        model_config,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Internal method to generate text using LiteLLM.
        """
        model_name = model_config["model_name"]

        __call_args = model_config.get("__call_args", {})
        for k, v in __call_args.items():
            if k not in kwargs:
                kwargs[k] = v
        
        # Handle O-series models which have specific parameter requirements
        model_id = f"{model_config['provider']}/{model_config['model']}"
        if any(o_model in model_id.lower() for o_model in ['o1', 'o2', 'o3', 'o4']):
            kwargs['temperature'] = 1.0
            kwargs['drop_params'] = True
        
        if messages is None:
            if prompt is None:
                raise ValueError("Either prompt or messages must be provided")
            messages = [{"role": "user", "content": prompt}]
        
        try:
            litellm_credentials = model_config["credentials"].copy()
            if model_config['provider'] == 'openrouter' and 'base_url' in litellm_credentials:
                litellm_credentials.pop('base_url', None)
            
            response = litellm.completion(
                model=model_id,
                messages=messages,
                **litellm_credentials,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"LiteLLM generation failed for model {model_name}") from e
        
        text = response.choices[0].message.content
        return {
            "model_name": model_name,
            "provider": model_config["provider"],
            "model": model_config["model"],
            "text": text,
            "usage": response.usage.model_dump() if response.usage else None,
            "finish_reason": response.choices[0].finish_reason,
        }
    
    @weave.op()
    def _generate_hf(
        self,
        model_config,
        prompt: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Internal method to generate text using a Hugging Face model.
        """
        self._load_hf_model_and_tokenizer(model_config["model"])

        max_tokens = kwargs.get("max_tokens", None)
        max_length = None
        
        if max_tokens is not None:
            try:
                input_length = len(self.hf_tokenizer(prompt).input_ids)
                max_length = input_length + max_tokens
                
                model_max_length = getattr(self.hf_tokenizer, 'model_max_length', None)
                if model_max_length is not None and max_length > model_max_length:
                    raise ValueError(f"Requested max_length ({max_length}) exceeds model's maximum ({model_max_length})")
                    
            except Exception as e:
                raise ValueError(f"Error calculating max_length: {e}")

        try:
            input_ids = self.hf_tokenizer(prompt, return_tensors="pt").input_ids
            input_tokens = input_ids.shape[1]
            
            outputs = self.generator(
                prompt,
                max_length=max_length,
                do_sample=True,
                **kwargs,
            )
            generated = outputs[0]["generated_text"]
            
            new_text = generated[len(prompt):].strip()
            
            full_output_ids = self.hf_tokenizer(generated, return_tensors="pt").input_ids
            total_tokens = full_output_ids.shape[1]
            
            output_tokens = max(0, total_tokens - input_tokens)
            
            return {
                "model_name": model_config["model_name"],
                "provider": model_config["provider"],
                "model": model_config["model"],
                "text": new_text,
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            }
        except Exception as e:
            raise RuntimeError(f"HuggingFace generation failed for model {model_config['model']}: {e}")

    def _load_vllm_model(self, model_name, **vllm_kwargs):
        """Load vLLM model with proper error handling and cleanup."""
        with self._model_lock:
            if self.vllm_model_name == model_name and self.vllm_model is not None:
                return

            self._cleanup_vllm_model()

            try:
                print(f"Loading vLLM model: {model_name}")
                self.vllm_model_name = model_name
                self.vllm_model = LLM(model=model_name, **vllm_kwargs)
                print(f"✓ vLLM model loaded: {model_name}")
            except Exception as e:
                self._cleanup_vllm_model()
                raise RuntimeError(f"Failed to load vLLM model {model_name}: {e}")

    @weave.op()
    def _generate_vllm(
        self,
        model_config,
        prompt: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Internal method to generate text using vLLM.
        """
        vllm_load_kwargs = model_config.get("vllm_kwargs", {})

        self._load_vllm_model(model_config["model"], **vllm_load_kwargs)

        max_tokens = kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", 8000))
        temperature = kwargs.pop("temperature", 1.0)

        hf_only_params = ['truncation', 'padding', 'return_tensors', 'add_special_tokens']
        for param in hf_only_params:
            kwargs.pop(param, None)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        try:
            outputs = self.vllm_model.generate([prompt], sampling_params)

            output = outputs[0]
            generated_text = output.outputs[0].text

            # Get token counts
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)

            return {
                "model_name": model_config["model_name"],
                "provider": model_config["provider"],
                "model": model_config["model"],
                "text": generated_text,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "finish_reason": output.outputs[0].finish_reason,
            }
        except Exception as e:
            raise RuntimeError(f"vLLM generation failed for model {model_config['model']}: {e}")

    async def generate_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Asynchronous wrapper for generate using ThreadPoolExecutor.
        """
        if self._closed:
            raise RuntimeError("Generator has been closed")
            
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.generate(prompt, messages, model_name, **kwargs)
        )

    async def generate_batch_async(
        self,
        prompts: List[str],
        model_name: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of prompts concurrently.

        Args:
            prompts: List of input prompt strings.
            model_name: Model identifier.
            **kwargs: Additional model-specific arguments.

        Returns:
            A list of dicts containing generated text and metadata for each prompt.
        """
        if not prompts:
            return []
            
        tasks = [
            self.generate_async(p, messages=None, model_name=model_name, **kwargs) 
            for p in prompts
        ]
        return await asyncio.gather(*tasks)

    def list_available_models(self) -> Dict[str, List[str]]:
        """
        List all available models by provider.
        """
        raise NotImplementedError("This class supports almost all models via LiteLLM. You just need to set the model name in models.yaml and credentials in credentials.yaml.")

