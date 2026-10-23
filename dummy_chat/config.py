"""
config_loader.py
Loads extraction configuration from YAML file or environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def load_config_from_yaml(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file

    Search order:
    1. Provided config_path
    2. EXTRACTION_CONFIG_PATH environment variable
    3. ./extraction_config.yaml
    4. ./config/extraction_config.yaml
    5. Default configuration
    """
    # Search paths
    search_paths = []

    if config_path:
        search_paths.append(config_path)

    if os.getenv("EXTRACTION_CONFIG_PATH"):
        search_paths.append(os.getenv("EXTRACTION_CONFIG_PATH"))

    search_paths.extend([
        "extraction_config.yaml",
        "config/extraction_config.yaml",
        "../extraction_config.yaml",
    ])

    # Try to load from each path
    for path in search_paths:
        if Path(path).exists():
            try:
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from: {path}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}")

    logger.info("No configuration file found, using defaults")
    return {}


def load_config_from_env() -> dict:
    """Load configuration from environment variables"""
    config = {}

    # Request settings
    if os.getenv("EXTRACTION_MAX_RETRIES"):
        config.setdefault("request", {})["max_retries"] = int(
            os.getenv("EXTRACTION_MAX_RETRIES"))

    if os.getenv("EXTRACTION_TIMEOUT"):
        config.setdefault("request", {})["timeout"] = int(
            os.getenv("EXTRACTION_TIMEOUT"))

    if os.getenv("EXTRACTION_USER_AGENT"):
        config.setdefault("request", {})["user_agent"] = os.getenv(
            "EXTRACTION_USER_AGENT")

    # Validation settings
    if os.getenv("EXTRACTION_MIN_CONTENT_LENGTH"):
        config.setdefault("validation", {})["min_content_length"] = int(
            os.getenv("EXTRACTION_MIN_CONTENT_LENGTH"))

    # Method toggles
    if os.getenv("EXTRACTION_USE_NEWSPAPER"):
        config.setdefault("methods", {})["newspaper"] = os.getenv(
            "EXTRACTION_USE_NEWSPAPER").lower() == "true"

    if os.getenv("EXTRACTION_USE_TRAFILATURA"):
        config.setdefault("methods", {})["trafilatura"] = os.getenv(
            "EXTRACTION_USE_TRAFILATURA").lower() == "true"

    if os.getenv("EXTRACTION_USE_READABILITY"):
        config.setdefault("methods", {})["readability"] = os.getenv(
            "EXTRACTION_USE_READABILITY").lower() == "true"

    if os.getenv("EXTRACTION_USE_PLAYWRIGHT"):
        config.setdefault("methods", {})["playwright"] = os.getenv(
            "EXTRACTION_USE_PLAYWRIGHT").lower() == "true"

    if os.getenv("EXTRACTION_USE_JINA"):
        config.setdefault("methods", {})["jina_reader"] = os.getenv(
            "EXTRACTION_USE_JINA").lower() == "true"

    # Jina settings
    if os.getenv("JINA_READER_API_KEY"):
        config.setdefault("jina", {})["api_key"] = os.getenv(
            "JINA_READER_API_KEY")

    if os.getenv("JINA_API_URL"):
        config.setdefault("jina", {})["api_url"] = os.getenv("JINA_API_URL")

    return config


def merge_configs(*configs: dict) -> dict:
    """Merge multiple configuration dictionaries (later configs override earlier ones)"""
    result = {}
    for config in configs:
        for key, value in config.items():
            if isinstance(value, dict) and key in result:
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    return result


def apply_config_to_extraction_config(config_dict: dict):
    """
    Apply loaded configuration to ExtractionConfig
    Import and use this function at startup
    """
    from dummy_chat.dummy_extractors import update_config

    # Flatten nested structure for update_config
    updates = {}

    # Request settings
    if "request" in config_dict:
        updates.update({
            "max_retries": config_dict["request"].get("max_retries"),
            "timeout": config_dict["request"].get("timeout"),
            "user_agent": config_dict["request"].get("user_agent"),
        })

    # Validation settings
    if "validation" in config_dict:
        updates.update({
            "min_content_length": config_dict["validation"].get("min_content_length"),
            "min_paragraph_length": config_dict["validation"].get("min_paragraph_length"),
        })

    # Method toggles
    if "methods" in config_dict:
        updates.update({
            "try_newspaper": config_dict["methods"].get("newspaper"),
            "try_trafilatura": config_dict["methods"].get("trafilatura"),
            "try_readability": config_dict["methods"].get("readability"),
            "try_playwright": config_dict["methods"].get("playwright"),
            "try_jina_reader": config_dict["methods"].get("jina_reader"),
        })

    # Jina settings
    if "jina" in config_dict:
        updates.update({
            "jina_api_key": config_dict["jina"].get("api_key"),
            "jina_api_url": config_dict["jina"].get("api_url"),
        })

    # Playwright settings
    if "playwright" in config_dict:
        updates.update({
            "playwright_timeout": config_dict["playwright"].get("timeout"),
            "playwright_wait_until": config_dict["playwright"].get("wait_until"),
        })

    # Content selectors
    if "content_selectors" in config_dict:
        updates["content_selectors"] = config_dict["content_selectors"]

    # UI blocklist
    if "ui_blocklist" in config_dict:
        updates["ui_text_blocklist"] = config_dict["ui_blocklist"]

    # Remove None values
    updates = {k: v for k, v in updates.items() if v is not None}

    # Apply updates
    if updates:
        update_config(**updates)
        logger.info(f"Applied {len(updates)} configuration updates")


def initialize_extraction_config(config_path: Optional[str] = None):
    """
    Initialize extraction configuration at application startup
    Call this in your FastAPI startup event or main()

    Priority order (later overrides earlier):
    1. Default configuration (in ExtractionConfig)
    2. YAML file configuration
    3. Environment variable configuration
    """
    yaml_config = load_config_from_yaml(config_path)
    env_config = load_config_from_env()

    # Merge configs (env overrides yaml)
    final_config = merge_configs(yaml_config, env_config)

    # Apply to ExtractionConfig
    apply_config_to_extraction_config(final_config)

    logger.info("Extraction configuration initialized successfully")


# Example usage in FastAPI
"""
from fastapi import FastAPI
from dummy_chat.config_loader import initialize_extraction_config

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    initialize_extraction_config()
    # Your other startup code...
"""
