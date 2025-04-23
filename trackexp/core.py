"""
Core functionality for trackexp.

This module provides the main functions for experiment tracking:
- init: Initialize a new experiment
- log: Log data for the experiment
- metadata: Store experiment configuration
"""

import pdb
import os
import json
import sqlite3
import datetime
import pickle
import shutil
import hashlib # <-- Import hashlib

from typing import Any, Callable, Dict, Optional, Union, Hashable
from pathlib import Path
from humanhash import humanize # <-- Import humanize

from .utils import get_experiment_path, ensure_dir_exists

# Global variables to track the current experiment
_current_experiment = None
_db_connection = None

# Dictionary for storing user variables
saved_vars = {}

def init(
    experiment_name: Optional[str] = None,
    base_dir: str = "experiments",
    humanhash_words: int = 4,
    overwrite: bool = True
) -> str:
    """
    Initialize a new experiment with either a provided name or a human-readable hash name.

    Args:
        experiment_name: Optional custom name for the experiment. If provided, uses this name
                         instead of generating a hash-based name.
        base_dir: Base directory for all experiments.
        humanhash_words: Number of words to use for the human-readable hash name (if auto-generating).
        overwrite: If True and experiment_name is provided, deletes any existing directory with that name.

    Returns:
        Path to the experiment directory.
    """
    global _current_experiment, _db_connection

    # If no custom name is provided, generate a hash-based name
    if experiment_name is None:
        # Generate a hash-based name as before
        timestamp = datetime.datetime.now().strftime("%Y%m%dc%H%M%Sc%f")
        hasher = hashlib.sha256()
        hasher.update(timestamp.encode('utf-8'))
        digest = hasher.hexdigest()

        human_hash_name = humanize(digest, words=humanhash_words)
        experiment_name = f"exp_{human_hash_name}"

        # Check for collisions with the generated name
        experiment_path = os.path.join(base_dir, experiment_name)
        if os.path.exists(experiment_path):
            print(f"Warning: Experiment directory {experiment_path} already exists. Appending timestamp.")
            experiment_name = f"{experiment_name}_{timestamp}"
            experiment_path = os.path.join(base_dir, experiment_name)
    else:
        # Use the provided name
        experiment_path = os.path.join(base_dir, experiment_name)

        # If directory exists and overwrite is True, delete it
        if os.path.exists(experiment_path):
            if overwrite:
                print(f"Removing existing experiment directory: {experiment_path}")
                shutil.rmtree(experiment_path)
            else:
                raise FileExistsError(f"Experiment directory {experiment_path} already exists and overwrite=False")

    # Create the experiment directory
    ensure_dir_exists(experiment_path)

    # Create artifacts directory
    artifacts_dir = os.path.join(experiment_path, "artifacts")
    ensure_dir_exists(artifacts_dir)

    # Store timestamp for metadata
    timestamp = datetime.datetime.now().strftime("%Y%m%dc%H%M%Sc%f")

    # Initialize SQLite database
    db_path = os.path.join(experiment_path, "experiment.db")
    conn = sqlite3.connect(db_path)

    # Store the experiment info
    _current_experiment = {
        "name": experiment_name,
        "path": experiment_path,
        "db_path": db_path,
        "artifacts_dir": artifacts_dir,
        "timestamp": timestamp,
        "human_hash": None if experiment_name else human_hash_name  # Only store hash if auto-generated
    }

    # Save the experiment info to a JSON file
    with open(os.path.join(experiment_path, "experiment_info.json"), "w") as f:
        json.dump(_current_experiment, f, indent=4)

    # Set the database connection
    _db_connection = conn

    print(f"Experiment '{experiment_name}' initialized at: {experiment_path}")
    return experiment_path


def _get_connection() -> sqlite3.Connection:
    """Get the SQLite connection for the current experiment."""
    global _db_connection
    if _db_connection is None:
        raise RuntimeError("No experiment initialized. Call trackexp.init() first.")
    return _db_connection

def _ensure_table_exists(context: str) -> None:
    """
    Ensure the table for the given context exists in the database.

    Args:
        context: The context name (will be used as table name).
    """
    conn = _get_connection()
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    # We use a flexible schema with JSON for storing different data types
    # identifier is stored as text to support various hashable types
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS "{context}" (
        identifier TEXT,
        name TEXT,
        value_type TEXT,
        value_data TEXT,
        timestamp REAL,
        PRIMARY KEY (identifier, name)
    )
    """)
    conn.commit()

def log(
    context: str,
    name: str,
    identifier: Hashable,
    data: Any,
    savefunc: Optional[Callable[[str, str, Hashable, Any], str]] = None
) -> None:
    """
    Log data for the current experiment.

    Args:
        context: The context (table) to log to.
        name: The name of the data point.
        identifier: A unique identifier for the row.
        data: The data to log.
        savefunc: Optional function to save data to disk.

    Example:
    trackexp.log("training", "loss", iter_index, loss_value)
                |          |         |                |
            context        |         |                |
         e.g.              |       "row id"           |
       "training"      name of     of tracked         |
     "validation"      tracked     data               |
   "testing"           data                     the data
                                                itself

    Note, you can store information inside
    trackexp.saved_vars = {}

    and use it like

    trackexp.saved_vars['iter'] = curr_iter
    """
    if _current_experiment is None:
        raise RuntimeError("No experiment initialized. Call trackexp.init() first.")

    # Convert identifier to string for storage
    identifier_str = str(identifier)

    # Ensure the table exists
    _ensure_table_exists(context)

    # Handle saving to disk if savefunc is provided
    if savefunc is not None:
        # Get the artifacts directory
        artifacts_dir = _current_experiment["artifacts_dir"]

        # Create context-specific subdirectory
        context_dir = os.path.join(artifacts_dir, context)
        ensure_dir_exists(context_dir)

        # Let the savefunc save the data and get the file path
        original_path = savefunc(context, name, identifier, data)

        # Extract just the filename
        filename = os.path.basename(original_path)

        # Create a path in our artifacts directory
        artifact_path = os.path.join(context_dir, filename)

        # If the file is not already in our artifacts directory, copy it there
        if os.path.abspath(original_path) != os.path.abspath(artifact_path):
            shutil.copy2(original_path, artifact_path)

        # Store the relative path from the experiment root
        rel_path = os.path.join("artifacts", context, filename)
        data = rel_path

    # Determine the type of data and serialize appropriately
    if isinstance(data, (int, float, bool, str, type(None))):
        value_type = type(data).__name__
        value_data = str(data)
    else:
        # For complex objects, pickle and store as base64
        value_type = "pickle"
        value_data = pickle.dumps(data).hex()

    # Store in the database
    conn = _get_connection()
    cursor = conn.cursor()

    # Record the current timestamp
    timestamp = datetime.datetime.now().timestamp()

    # Insert or replace existing data point
    cursor.execute(f"""
    INSERT OR REPLACE INTO "{context}"
    (identifier, name, value_type, value_data, timestamp)
    VALUES (?, ?, ?, ?, ?)
    """, (identifier_str, name, value_type, value_data, timestamp))

    conn.commit()

def metadata(config: Dict[str, Any]) -> None:
    """
    Store experiment metadata/configuration.

    Args:
        config: Dictionary of configuration parameters.
    """
    if _current_experiment is None:
        raise RuntimeError("No experiment initialized. Call trackexp.init() first.")

    # Save as special "metadata" context with a single identifier
    log("metadata", "config", "experiment", config)

    # Also save as a JSON file for easy access
    metadata_path = os.path.join(_current_experiment["path"], "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(config, f, indent=4)

def get_current_experiment() -> Optional[Dict[str, str]]:
    """
    Get information about the current experiment.

    Returns:
        Dictionary with experiment details or None if no experiment is initialized.
    """
    return _current_experiment
