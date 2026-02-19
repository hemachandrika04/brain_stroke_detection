"""Launcher: run the Flask app from project root (e.g. python app.py)."""
import os
import sys

# Run backend/app.py from the backend directory so paths and model file resolve correctly
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
os.chdir(backend_dir)
sys.path.insert(0, backend_dir)

# Load and run the app
from app import app  # noqa: E402

if __name__ == "__main__":
    app.run(debug=True)
