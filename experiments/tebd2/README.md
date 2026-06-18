# tebd2

Placeholder suite for TEBD² (imaginary-time ground-state) experiments. There are
no experiment scripts here yet.

The TEBD² imaginary-time ground-state solver itself lives in the package, not in
this directory, at:

```
rand_isopeps.real_isotns.tebd2
```

A back-compat import is also available:

```python
from rand_isopeps.tebd import tfi_ham, imaginary_time
```

The solver needs the optional `quimb` dependency (`pip install -e ".[quimb]"`).

## Status

Experiment drivers will be added under `scripts/` later, following the same run
pattern as the other suites:

```
pip install -e ".[quimb]"
python experiments/tebd2/scripts/<name>.py --quick
```

Outputs would land in `outputs/tebd2/` (gitignored).
