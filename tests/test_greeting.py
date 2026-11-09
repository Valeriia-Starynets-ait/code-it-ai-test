import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from greeting import greet


def test_greet():
    """Test that greet returns the correct greeting message."""
    assert greet("World") == "Hello, World!"
    assert greet("Alice") == "Hello, Alice!"
    assert greet("Bob") == "Hello, Bob!"
