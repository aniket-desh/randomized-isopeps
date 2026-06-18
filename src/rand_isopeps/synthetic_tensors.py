"""Back-compat shim. Canonical module: rand_isopeps.synthetic.tensors

This module was relocated during the src/ layout restructure. Importing
'rand_isopeps.synthetic_tensors' keeps working and re-exports every public name from
rand_isopeps.synthetic.tensors.
"""
import importlib as _il

_m = _il.import_module("rand_isopeps.synthetic.tensors")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("_")})
del _il, _m
