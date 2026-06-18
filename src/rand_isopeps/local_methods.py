"""Back-compat shim. Canonical module: rand_isopeps.moses.local_methods

This module was relocated during the src/ layout restructure. Importing
'rand_isopeps.local_methods' keeps working and re-exports every public name from
rand_isopeps.moses.local_methods.
"""
import importlib as _il

_m = _il.import_module("rand_isopeps.moses.local_methods")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("_")})
del _il, _m
