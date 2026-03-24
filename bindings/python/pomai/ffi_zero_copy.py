"""
Deprecated compatibility shim.

Use `pomaidb.zero_copy` (or `from pomaidb import search_zero_copy`).
"""

from pomaidb.zero_copy import release_zero_copy_session, search_zero_copy

__all__ = ["search_zero_copy", "release_zero_copy_session"]
