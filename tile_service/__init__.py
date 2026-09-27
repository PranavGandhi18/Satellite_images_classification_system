"""Offline land-use tile classification service (Part 2 working slice).

Runtime dependencies are deliberately small: ONNX Runtime, NumPy, Pillow, FastAPI and the standard library's sqlite3.
No PyTorch, no network access, no model downloads: everything the service needs is inside models/<version>/.
"""
__version__ = '0.1.0'
