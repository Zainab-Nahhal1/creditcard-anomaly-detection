
from src.main import load_data

def test_load_data():
    assert callable(load_data), "load_data should be a callable function."
