"""
Utility functions for trackexp.

This module provides helper functions for:
- Directory management
- Experiment discovery
- Data retrieval
"""

import os
import json
import sqlite3
import pickle
import ast
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

def ensure_dir_exists(dir_path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        dir_path: Path to the directory.
    """
    os.makedirs(dir_path, exist_ok=True)

def get_experiment_path(experiment_name: str, base_dir: str = "trackexp_out") -> str:
    """
    Get the path to an experiment directory.

    Args:
        experiment_name: Name of the experiment.
        base_dir: Base directory for all experiments.

    Returns:
        Path to the experiment directory.

    Raises:
        FileNotFoundError: If the experiment doesn't exist.
    """
    path = os.path.join(base_dir, experiment_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Experiment '{experiment_name}' not found in '{base_dir}'")
    return path

def list_experiments(base_dir: str = "trackexp_out") -> List[Dict[str, Any]]:
    """
    List all experiments in the base directory.

    Args:
        base_dir: Base directory for all experiments.

    Returns:
        List of dictionaries with experiment information.
    """
    if not os.path.isdir(base_dir):
        return []

    experiments = []

    for item in os.listdir(base_dir):
        exp_dir = os.path.join(base_dir, item)
        if os.path.isdir(exp_dir):
            info_path = os.path.join(exp_dir, "experiment_info.json")
            if os.path.isfile(info_path):
                try:
                    with open(info_path, "r") as f:
                        info = json.load(f)
                    experiments.append(info)
                except (json.JSONDecodeError, IOError):
                    # Skip if we can't load the info
                    pass

    # Sort by timestamp, newest first
    experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return experiments

def parse_value(value_type: str, value_data: str) -> Any:
    """
    Parse a value from the database based on its type.

    Args:
        value_type: Type of the value.
        value_data: String representation of the value.

    Returns:
        The parsed value.
    """
    if value_type == "int":
        return int(value_data)
    elif value_type == "float":
        return float(value_data)
    elif value_type == "bool":
        return value_data.lower() == "true"
    elif value_type == "str":
        return value_data
    elif value_type == "NoneType":
        return None
    elif value_type == "pickle":
        return pickle.loads(bytes.fromhex(value_data))
    else:
        # Default to returning the raw string
        return value_data

def get_data(
    experiment_name: str,
    context: str,
    name: Optional[str] = None,
    identifier: Optional[str] = None,
    base_dir: str = "trackexp_out",
    wide: bool = True,
    sort_by_ident: bool = True
) -> Union[pd.DataFrame, Dict[Tuple[str, str], Any]]:
    """
    Get data from an experiment.

    Args:
        experiment_name: Name of the experiment.
        context: The context (table) to query.
        name: Optional name filter.
        identifier: Optional identifier filter.
        base_dir: Base directory for all experiments.
        wide: Whether to return a wide dataframe (pivot table) or regular dataframe.

    Returns:
        Either a wide DataFrame or a regular DataFrame based on the 'wide' parameter.
    """
    if os.path.isabs(experiment_name) or os.path.sep in experiment_name:
        # Treat as a path directly
        exp_path = os.path.abspath(experiment_name)
    else:
        # Treat as folder name
        exp_path = get_experiment_path(experiment_name, base_dir)

    db_path = os.path.join(exp_path, "experiment.db")

    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"Database not found for experiment '{experiment_name}'")

    conn = sqlite3.connect(db_path)

    # Get both value_data and value_type
    query = f"SELECT identifier, name, value_type, value_data, timestamp FROM {context}"

    # Add filters if provided
    conditions = []
    params = []
    if name is not None:
        conditions.append("name = ?")
        params.append(name)
    if identifier is not None:
        conditions.append("identifier = ?")
        params.append(str(identifier))

    if conditions:
        query += f" WHERE {' AND '.join(conditions)}"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Deserialize the data based on its value_type
    def deserialize_value(row):
        value_type = row['value_type']
        value_data = row['value_data']

        if value_type == 'pickle':
            # Convert hex string back to bytes and unpickle
            return pickle.loads(bytes.fromhex(value_data))
        elif value_type in ('int', 'float', 'bool'):
            # Convert string to the appropriate type
            type_map = {'int': int, 'float': float, 'bool': lambda x: x.lower() == 'true'}
            return type_map[value_type](value_data)
        else:
            # Return strings and other types as is
            return value_data

    # Apply the deserialize function to create a new column with the actual values
    df['value'] = df.apply(deserialize_value, axis=1)

    if not wide:
        return df

    # For wide format, pivot using the deserialized values
    wide_df = df.pivot(index='identifier', columns='name', values='value')

    wide_df = wide_df.reset_index()

    if sort_by_ident:
        wide_df['_identifier'] = wide_df['identifier'].apply(lambda x: ast.literal_eval(x))

        # Sort the DataFrame by the tuple, ensuring the second element increases fastest
        wide_df = wide_df.sort_values(by='_identifier')

        if 'elapsed_time' in set(wide_df.columns):
            wide_df['wallclocktime'] = wide_df['elapsed_time'].cumsum()

    # Reset the index to make 'identifier' a regular column
    wide_df = wide_df.reset_index()


    return wide_df

def get_metadata(experiment_name: str, base_dir: str = "trackexp_out") -> Dict[str, Any]:
    """
    Get metadata for an experiment.

    Args:
        experiment_name: Name of the experiment.
        base_dir: Base directory for all experiments.

    Returns:
        Metadata dictionary.
    """
    # Try to load from metadata.json first (faster)
    exp_path = get_experiment_path(experiment_name, base_dir)
    metadata_path = os.path.join(exp_path, "metadata.json")

    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Fall back to database
    data = get_data(experiment_name, "metadata", "config", "experiment", base_dir)
    if data:
        return data.get(("experiment", "config"), {})

    return {}
