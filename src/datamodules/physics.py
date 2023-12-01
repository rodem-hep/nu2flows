"""A collection of functions for particle physics calulations."""
from __future__ import annotations  # Needed for type hinting itself

import numpy as np
import torch as T

from mltools.mltools.torch_utils import empty_0dim_like


class Mom4Vec:
    """A simple class that contains the four vectors of events in either px, py, pz, E
    or pt, eta, phi, E.

    Will automatically work with both numpy arrays and pytorch tensors
    """

    def __init__(
        self,
        data: np.ndarray | T.Tensor,
        oth: None | np.ndarray | T.Tensor = None,
        is_cartesian: bool = True,
        final_is_mass: bool = False,
        n_mom: int | None = None,
    ) -> None:
        """
        args:
            data: The input array where the last dimension points to the coordinates
            oth: Other data which is not used as part of the momentum
        kwargs:
            is_cartesian: If the data provided is wrt cartesian coordinates
                True:  self.mom = [px, py, px, E or M]
                False: self.mom = [pt, eta, phi, E or M]
            final_is_mass:
                If true then the 4th variable in the tensor is taken to be mass instead
                of energy
            n_mom: How many variables in the list refer to momentum
                If none it infers from the shape
        """

        # Save the attributes
        self.is_cartesian = is_cartesian
        if isinstance(data, T.Tensor):
            self.is_tensor = True
        elif isinstance(data, np.ndarray):
            self.is_tensor = False
        else:
            raise ValueError(
                "Mom4Vec is not able to tell if data is a torch tensor "
                "or a numpy array!"
            )

        # Create the 4 momentum array/tensor
        if self.is_tensor:
            self.mom = T.zeros((*data.shape[:-1], 4), dtype=T.float32)
        else:
            self.mom = np.zeros((*data.shape[:-1], 4), dtype=np.float32)

        # Infer the number of kinematic inputs
        if n_mom is None:
            n_mom = min(data.shape[-1], 4)

        # Fill in the momentum array
        if n_mom == 4:
            self.mom = data[..., :4].clone() if self.is_tensor else data[..., :4].copy()
            if final_is_mass:
                self.mom[..., 3:4] = (
                    T.sqrt(self.p3_mag**2 + self.mom[..., 3:4] ** 2)
                    if self.is_tensor
                    else np.sqrt(self.p3_mag**2 + self.mom[..., 3:4] ** 2)
                )

        elif n_mom == 3:
            self.mom[..., :3] = data[..., :3]
            self.set_massless_energy()
        elif n_mom == 2:
            if self.is_cartesian:
                self.mom[..., :2] = data[..., :2]
                self.set_massless_energy()
            else:
                self.mom[..., 0:1] = data[..., 0:1]
                self.mom[..., 2:3] = data[..., 1:2]
                self.set_massless_energy()

        # The leftover tensor which holds the rest of the variables
        if oth is None:
            self.oth = data[..., n_mom:]
        else:
            if self.is_tensor:
                self.oth = T.cat([data[..., n_mom:], oth], dim=-1)
            else:
                self.oth = np.concatenate([data[..., n_mom:], oth], axis=-1)

    @property
    def shape(self) -> tuple:
        return self.mom.shape

    @property
    def pt(self) -> np.ndarray | T.Tensor:
        """Return the transverse momentum."""
        if self.is_cartesian:
            pt = np.linalg.norm(self.mom[..., :2], axis=-1, keepdims=True)
            if self.is_tensor:
                pt = T.from_numpy(pt)
            return pt
        else:
            return self.mom[..., 0:1]

    @property
    def eta(self) -> np.ndarray | T.Tensor:
        """Return the pseudorapitity."""
        if self.is_cartesian:
            return np.arctanh(
                np.clip((self.pz / (self.p3_mag + 1e-8)), 1e-6 - 1, 1 - 1e-6)
            )
        else:
            return self.mom[..., 1:2]

    @property
    def phi(self) -> np.ndarray | T.Tensor:
        """Return the polar angle."""
        if self.is_cartesian:
            return np.arctan2(self.py, self.px)
        else:
            return self.mom[..., 2:3]

    @property
    def px(self) -> np.ndarray | T.Tensor:
        """Return the momentum x component."""
        if self.is_cartesian:
            return self.mom[..., 0:1]
        else:
            return self.pt * np.cos(self.phi)

    @property
    def py(self) -> np.ndarray | T.Tensor:
        """Return the momentum y component."""
        if self.is_cartesian:
            return self.mom[..., 1:2]
        else:
            return self.pt * np.sin(self.phi)

    @property
    def pz(self) -> np.ndarray | T.Tensor:
        """Return the momentum z component."""
        if self.is_cartesian:
            return self.mom[..., 2:3]
        else:
            return self.pt * np.sinh(self.eta)

    @property
    def p3_mag(self) -> np.ndarray | T.Tensor:
        """Return the magnitude of the 3 momentum."""
        if self.is_cartesian:
            p3_mag = np.linalg.norm(self.mom[..., :3], axis=-1, keepdims=True)
            if self.is_tensor:
                p3_mag = T.from_numpy(p3_mag)
            return p3_mag
        else:
            return self.pt * np.cosh(self.eta)

    @property
    def energy(self) -> np.ndarray | T.Tensor:
        """Return the energy."""
        return self.mom[..., 3:4]

    @property
    def E(self) -> np.ndarray | T.Tensor:
        """Shorthand for energy."""
        return self.energy

    @property
    def theta(self) -> np.ndarray | T.Tensor:
        """Shorthand for energy."""
        return 2 * np.arctan(np.exp(-self.eta))

    @property
    def mass(self) -> np.ndarray | T.Tensor:
        """Return the mass using the m2=E2-p2."""
        if self.is_cartesian:
            return np.sqrt(
                np.abs(self.energy**2 - (self.px**2 + self.py**2 + self.pz**2))
            )
        else:
            mass = np.sqrt(
                np.abs(self.energy**2 - (self.pt * np.cosh(self.eta)) ** 2)
            )
            return mass

    @property
    def m(self) -> np.ndarray | T.Tensor:
        """Shorthand for mass."""
        return self.mass

    @property
    def m2(self) -> np.ndarray | T.Tensor:
        """Shorthand for mass squared."""
        return self.mass**2

    @property
    def beta(self) -> np.ndarray | T.Tensor:
        """Return the beta value."""
        return self.p3_mag / self.E

    @property
    def rapidity(self) -> np.ndarray | T.Tensor:
        """Return the beta value."""
        return np.log((self.E + self.pz) / (self.E - self.pz)) / 2

    def __len__(self):
        return len(self.mom)

    def apply_mask(self, mask: np.ndarray) -> None:
        """Apply a mask to both the momentum and the other arrays."""
        self.mom = self.mom[mask]
        self.oth = self.oth[mask]

    def set_massless_energy(self) -> None:
        """Sets the energy part of the 4 momentum tensor to be the 3 momentum mag."""
        self.mom[..., 3:4] = self.p3_mag

    def to_cartesian(self) -> None:
        """Changes the saved momentum tensor to cartesian inplace."""
        if self.is_cartesian:
            return

        else:
            px = self.px.clone() if self.is_tensor else self.px.copy()
            py = self.py.clone() if self.is_tensor else self.py.copy()
            pz = self.pz.clone() if self.is_tensor else self.pz.copy()
            self.mom[..., 0:1] = px
            self.mom[..., 1:2] = py
            self.mom[..., 2:3] = pz
            self.is_cartesian = True

    def to_spherical(self) -> None:
        """Changes the saved momentum tensor to spherical inplace."""
        if not self.is_cartesian:
            return

        else:
            pt = self.pt.clone() if self.is_tensor else self.pt.copy()
            eta = self.eta.clone() if self.is_tensor else self.eta.copy()
            phi = self.phi.clone() if self.is_tensor else self.phi.copy()
            self.mom[..., 0:1] = pt
            self.mom[..., 1:2] = eta
            self.mom[..., 2:3] = phi
            self.is_cartesian = False

    def __add__(self, other: Mom4Vec) -> Mom4Vec:
        """Add two collections of four momentum together."""
        assert self.is_cartesian and other.is_cartesian
        return Mom4Vec(self.mom + other.mom, is_cartesian=True)

    def __sub__(self, other: Mom4Vec) -> Mom4Vec:
        """Subtract two collections of four momentum together."""
        assert self.is_cartesian and other.is_cartesian
        return Mom4Vec(self.mom - other.mom, is_cartesian=True)

    def __mul__(self, a: Mom4Vec | float) -> float | Mom4Vec:
        """Multiply by a float or another 4 vector."""
        # Multiply the momentum values by a scalar
        if isinstance(a, (float, int)):
            px = self.px * a
            py = self.py * a
            pz = self.pz * a
            m = self.m
            E = np.sqrt(np.abs(m**2 - (px**2 + py**2 + pz**2)))
            mom = np.concatenate([px, py, pz, E], axis=-1)
            return Mom4Vec(mom, oth=self.oth)

        # Lotentz dot product
        if isinstance(a, Mom4Vec):
            return self.E * a.E - self.px * a.px - self.py * a.py - self.pz * a.pz

    def __getitem__(self, idx: int | np.ndarray | slice) -> Mom4Vec:
        """Index, mask or slice the object."""
        if isinstance(idx, int):
            if idx == -1:
                idx = slice(idx, None)
            else:
                idx = slice(idx, idx + 1)
        return Mom4Vec(self.mom[idx], self.oth[idx], is_cartesian=self.is_cartesian)

    def __repr__(self) -> str:
        return f"Mom4Vec({self.mom.shape})"


def delR(x: Mom4Vec, y: Mom4Vec) -> np.ndarray:
    return np.sqrt((x.eta - y.eta) ** 2 + (x.phi - y.phi) ** 2)


def change_from_ptetaphiE(
    data: np.ndarray, old_cords: list, new_cords: list, n_dim: int = 0
) -> tuple[np.ndarray, list]:
    """Converts a tensor from spherical to a new set of coordinates.

    Makes assumptions on based on number of features in final dimension if n_dim is
    not explictly set
    - 2D = pt, phi
    - 3D = pt, eta, phi
    - 4D = pt, eta, phi, energy
    - 4D+ = pt, eta, phi, energy, other... (other is not changed, always kept)

    Args:
        data: A multidimensional tensor containing the sperical components
        old_names: The current names of the coords
        new_names: The new coords to calculate
        n_dim: The number of dimensions to transform, if 0 it will be assumed

    Returns:
        new_values, new_names
    """

    # Allow a string to be given which can be seperated into a list
    old_names = old_cords.split(",") if isinstance(old_cords, str) else old_cords
    new_names = new_cords.split(",") if isinstance(new_cords, str) else new_cords

    # List of supported new names
    for new_nm in new_names:
        if new_nm not in [
            "pt",
            "log_pt",
            "energy",
            "log_energy",
            "phi",
            "cos",
            "sin",
            "eta",
            "px",
            "py",
            "pz",
        ]:
            raise ValueError(f"Unknown coordinate name: {new_nm}")

    # Calculate the number of kinematic features in the final dimension
    n_dim = n_dim or min(4, data.shape[-1])

    # Create the slices
    pt = data[..., 0:1]
    eta = data[..., 1:2] if n_dim > 2 else empty_0dim_like(data)
    phi = data[..., 2:3] if n_dim > 2 else data[..., 1:2]
    eng = data[..., 3:4] if n_dim > 3 else empty_0dim_like(data)
    oth = data[..., n_dim:]

    # If energy is empty then we might try use pt and eta
    if eng.shape[-1] == 0 and eta.shape[-1] > 0:
        eng = pt * np.cosh(eta)

    # A dictionary for calculating the supported new variables
    # Using lambda functions like this prevents execution every time
    new_coord_fns = {
        "pt": lambda: pt,
        "log_pt": lambda: np.log(pt + 1e-8),
        "energy": lambda: eng,
        "log_energy": lambda: np.log(eng + 1e-8),
        "phi": lambda: phi,
        "cos": lambda: np.cos(phi),
        "sin": lambda: np.sin(phi),
        "eta": lambda: eta,
        "px": lambda: pt * np.cos(phi),
        "py": lambda: pt * np.sin(phi),
        "pz": lambda: pt * np.sinh(eta),
        "oth": lambda: oth,
    }

    # Create a dictionary of the requested coordinates then trim non empty values
    new_coords = {key: new_coord_fns[key]() for key in new_names}
    new_coords = {key: val for key, val in new_coords.items() if val.shape[-1] != 0}

    # Return the combined tensors and the collection of new names (with unchanged)
    new_vals = np.concatenate(list(new_coords.values()) + [oth], axis=-1)
    new_names = list(new_coords.keys())
    new_names += old_names[n_dim:]

    return new_vals, new_names


def nu_sol_comps(
    lept_px: np.ndarray,
    lept_py: np.ndarray,
    lept_pz: np.ndarray,
    lept_e: np.ndarray,
    lept_ismu: np.ndarray,
    nu_px: np.ndarray,
    nu_py: np.ndarray,
) -> np.ndarray:
    """Calculate the components of the quadratic solution for neutrino
    pseudorapidity."""

    # Constants NuRegData is in GeV!
    w_mass = 80379 * 1e-3
    e_mass = 0.511 * 1e-3
    mu_mass = 105.658 * 1e-3

    # Create an array of the masses using lepton ID
    l_mass = np.where(lept_ismu != 0, mu_mass, e_mass)

    # Calculate all components in the quadratic equation
    nu_ptsq = nu_px**2 + nu_py**2
    alpha = w_mass**2 - l_mass**2 + 2 * (lept_px * nu_px + lept_py * nu_py)
    a = lept_pz**2 - lept_e**2
    b = alpha * lept_pz
    # c = alpha**2 / 4 - lept_e**2 * nu_ptsq # Using delta instead (quicker)
    delta = lept_e**2 * (alpha**2 + 4 * a * nu_ptsq)

    comp_1 = -b / (2 * a)
    comp_2 = delta / (4 * a**2)

    # Take the sign preserving sqrt of comp_2 due to scale
    comp_2 = np.sign(comp_2) * np.sqrt(np.abs(comp_2))
    return comp_1, comp_2


def combine_comps(
    comp_1: np.ndarray,
    comp_2: np.ndarray,
    return_eta: bool = False,
    nu_pt: np.ndarray = None,
    return_both: bool = False,
) -> np.ndarray:
    """Combine the quadiratic solutions and pick one depending on complexity and size
    args:
        comp_1: First component of the quadratic
        comp_2: Signed root of the second component of the quadratic
    kwargs:
        return_eta: If the output should be eta, otherwise pz
        nu_pt: The neutrino pt, needed only if return_eta is true
        return_both: Return both solutions
    """

    # comp_2 is already rooted, so the real component is taken to be 0 if negative
    comp_2_real = np.where(comp_2 > 0, comp_2, np.zeros_like(comp_2))

    # Get the two solutions
    sol_1 = comp_1 + comp_2_real
    sol_2 = comp_1 - comp_2_real

    # If both solutions are requested
    if return_both:
        if return_eta:
            return (
                np.arctanh(sol_1 / np.sqrt(sol_1**2 + nu_pt**2 + 1e-8)),
                np.arctanh(sol_2 / np.sqrt(sol_2**2 + nu_pt**2 + 1e-8)),
            )
        return sol_1, sol_2

    # Take the smallest solution based on magnitude
    sol = np.where(np.abs(sol_1) < np.abs(sol_2), sol_1, sol_2)

    # Return correct variable
    if return_eta:
        return np.arctanh(sol / np.sqrt(sol**2 + nu_pt**2 + 1e-8))
    else:
        return sol
