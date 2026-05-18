"""Runs a machine learning model based on EMBER-2 for emulating cosmological simulations.

Run the whole module with `python -m cs_5960_ast {train,infer}` (which invokes __main__.py) or run entry_point.py manually.
Or use uv as specified in README.
"""

from cs_5960_ast.entry_point import main

if __name__ == "__main__":
    main()
