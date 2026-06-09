"""dimension bookkeeping for synthetic block Moses Move kernels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MosesDims:
    """Grouped local dimensions used by the block sequential Moses Move.

    The excited-state block-isoPEPS paper groups an interior active tensor as
    B in C^(n1 x n2 x n3) with

        n1 = chi * eta * d,
        n2 = eta * p,
        n3 = chi * eta.

    The two local truncation ranks are k1 = chi * eta and k2 = eta.
    """

    chi: int
    eta: int
    d: int = 2
    p: int = 1

    def __post_init__(self) -> None:
        for name in ("chi", "eta", "d", "p"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def n1(self) -> int:
        return self.chi * self.eta * self.d

    @property
    def n2(self) -> int:
        return self.eta * self.p

    @property
    def n3(self) -> int:
        return self.chi * self.eta

    @property
    def k1(self) -> int:
        return self.chi * self.eta

    @property
    def k2(self) -> int:
        return self.eta

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        return (self.n1, self.n2, self.n3)

    @property
    def first_svd_shape(self) -> tuple[int, int]:
        return (self.n1, self.n2 * self.n3)

    @property
    def second_svd_shape(self) -> tuple[int, int]:
        return (self.eta * self.n2, self.chi * self.n3)

    @property
    def first_svd_cost_model(self) -> int:
        return self.n1 * self.n1 * self.n2 * self.n3

    @property
    def second_svd_cost_model(self) -> int:
        return (self.eta * self.n2) ** 2 * self.chi * self.n3

    @property
    def local_cost_model(self) -> int:
        return self.first_svd_cost_model + self.second_svd_cost_model

    def as_dict(self) -> dict[str, int]:
        return {
            "chi": self.chi,
            "eta": self.eta,
            "d": self.d,
            "p": self.p,
            "n1": self.n1,
            "n2": self.n2,
            "n3": self.n3,
            "k1": self.k1,
            "k2": self.k2,
            "first_rows": self.first_svd_shape[0],
            "first_cols": self.first_svd_shape[1],
            "second_rows": self.second_svd_shape[0],
            "second_cols": self.second_svd_shape[1],
            "first_cost_model": self.first_svd_cost_model,
            "second_cost_model": self.second_svd_cost_model,
            "local_cost_model": self.local_cost_model,
        }


def dims_grid(
    chis: list[int],
    etas: list[int],
    ps: list[int],
    d: int = 2,
) -> list[MosesDims]:
    return [MosesDims(chi=chi, eta=eta, d=d, p=p) for chi in chis for eta in etas for p in ps]
