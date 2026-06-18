"""Back-compat shim. Canonical module: rand_isopeps.linalg.randomized_svd

This module was relocated during the src/ layout restructure. Importing
'rand_isopeps.randomized_svd' keeps working and re-exports every public name from
rand_isopeps.linalg.randomized_svd.
"""
import importlib as _il

_m = _il.import_module("rand_isopeps.linalg.randomized_svd")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("_")})
del _il, _m
