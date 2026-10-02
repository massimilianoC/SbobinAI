"""Compatibility re-exports for JSON filesystem operations."""

from ..application.state import atomic_json, read_json

__all__ = ["atomic_json", "read_json"]
