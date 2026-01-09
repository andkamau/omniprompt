import os
from pathlib import Path
from omniprompt import save_image, get_fun_caption, FUN_CAPTIONS

def test_get_fun_caption():
    caption = get_fun_caption()
    assert caption in FUN_CAPTIONS
    assert isinstance(caption, str)

def test_save_image(tmp_path, monkeypatch):
    # Patch the global GENERATED_IMAGES_DIR in the module to use the temp path
    monkeypatch.setattr("omniprompt.GENERATED_IMAGES_DIR", tmp_path)
    
    dummy_data = b"\x89PNG\r\n\x1a\n"
    provider = "test_provider"
    prompt = "A test prompt with special chars! @#"
    
    filepath = save_image(dummy_data, provider, prompt)
    
    # Check if file exists
    assert filepath.exists()
    assert filepath.read_bytes() == dummy_data
    
    # Check filename structure
    filename = filepath.name
    assert filename.startswith(f"{provider}_")
    assert filename.endswith(".png")
    # Check sanitization (spaces to underscores, remove special chars)
    assert "A_test_prompt_with_special_chars" in filename
    assert "@" not in filename
