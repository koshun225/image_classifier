"""
Helpers for parameter schema that mixes fixed values and Optuna definitions.
"""
from __future__ import annotations

from typing import Any, Dict, List


def materialize_params(schema: Any) -> Any:
    """
    Convert schema nodes that contain Optuna metadata into plain values.

    Nodes with {"value": X, "type": "..."} or {"value": X} are replaced by X.
    """

    if isinstance(schema, dict):
        # valueキーがある場合は、typeの有無に関わらずvalueを返す
        if "value" in schema:
            return schema.get("value")
        # valueキーがない場合は再帰的に処理
        return {key: materialize_params(value) for key, value in schema.items()}

    if isinstance(schema, list):
        return [materialize_params(item) for item in schema]

    return schema


def extract_tunable_specs(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Walk through schema and return mapping of dotted paths -> spec dicts
    for nodes that contain Optuna metadata.
    """

    specs: Dict[str, Dict[str, Any]] = {}

    def _recurse(node: Any, path: List[str]) -> None:
        if isinstance(node, dict):
            if "type" in node and "value" in node:
                specs[".".join(path)] = node
            else:
                for key, value in node.items():
                    _recurse(value, path + [key])
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                _recurse(value, path + [str(idx)])

    _recurse(schema, [])
    return specs


def set_nested_value(target: Dict[str, Any], path: List[str], value: Any) -> None:
    """
    Set nested value inside dict using path list, creating intermediate dicts.
    """

    cursor = target
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value

