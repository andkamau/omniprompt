"""
Utility functions and constants for OmniPrompt.
"""

import os
import random
import requests
import base64
import concurrent.futures
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# --- Constants ---
GENERATED_IMAGES_DIR = Path("generated_images")

FUN_CAPTIONS = [
    "Convincing the pixels to cooperate...",
    "Mixing red, green, and blue in a cauldron...",
    "Consulting the oracle of aesthetics...",
    "Teaching the AI art history in 5 seconds...",
    "Dreaming in electric sheep...",
    "Summoning the muse from the cloud...",
    "Applying virtual paint to digital canvas...",
    "Negotiating with the GPU...",
    "Connecting the dots... all million of them...",
    "Polishing the pixels...",
]

def get_fun_caption() -> str:
    """Returns a random fun caption for display during long operations."""
    return random.choice(FUN_CAPTIONS)

def save_image(data: bytes, provider: str, prompt: str, extension: str = "png") -> Path:
    """
    Saves image data (bytes) to the generated_images directory.

    Args:
        data: Binary image data.
        provider: The name of the API provider.
        prompt: The original prompt used to generate the image (for filename).
        extension: File extension (default "png").

    Returns:
        The Path to the saved image file.
    """
    GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create a filename based on timestamp and prompt (sanitized)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_prompt = "".join(x for x in prompt if x.isalnum() or x in " -_")[:50].strip().replace(" ", "_")
    filename = f"{provider}_{timestamp}_{sanitized_prompt}.{extension}"
    filepath = GENERATED_IMAGES_DIR / filename
    
    with open(filepath, "wb") as f:
        f.write(data)
    
    return filepath

def download_image(url: str) -> bytes:
    """
    Downloads an image from a URL.

    Args:
        url: The URL of the image.

    Returns:
        The binary content of the image.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.content

def run_with_dynamic_captions(console: Console, action: Callable, *args, **kwargs) -> Any:
    """
    Runs an action in a separate thread while updating UI captions.

    Args:
        console: The Rich console instance.
        action: The function to execute.
        *args: Positional arguments for the action.
        **kwargs: Keyword arguments for the action.

    Returns:
        The result of the action function.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task(get_fun_caption(), total=None)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(action, *args, **kwargs)
            
            while not future.done():
                time.sleep(2.0)  # Change caption every 2 seconds
                if not future.done():
                    progress.update(task, description=get_fun_caption())
            
            return future.result()
