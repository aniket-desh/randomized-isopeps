"""Back-compat shim. Canonical module: rand_isopeps.experiment_utils.io

This module was relocated during the src/ layout restructure. Importing
'rand_isopeps.io_utils' keeps working and re-exports every public name from
rand_isopeps.experiment_utils.io.
"""
import importlib as _il

_m = _il.import_module("rand_isopeps.experiment_utils.io")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("_")})
del _il, _m
