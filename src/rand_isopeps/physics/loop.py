"""the single fixed-iteration loop used by every phase-two experiment."""

from __future__ import annotations


def run_iterations(state, *, iterations: int, update, measure, on_record=None):
    """run ``state = update(state, k)`` exactly ``iterations`` times.

    ``update`` returns ``(state, update_metrics)`` and ``measure`` returns a
    serializable dictionary. The history includes iteration zero.
    """
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    history = [dict(iteration=0, **measure(state, 0))]
    if on_record is not None:
        on_record(history[-1])
    for iteration in range(1, int(iterations) + 1):
        state, update_metrics = update(state, iteration)
        history.append(dict(
            iteration=iteration,
            **measure(state, iteration),
            **update_metrics,
        ))
        if on_record is not None:
            on_record(history[-1])
    return state, history
