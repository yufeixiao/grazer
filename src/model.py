"""
Spatial Grazer-Vegetation Model

Minimal 2-equation reaction-diffusion system for Tibetan Plateau grasslands.

Model equations:
  ∂V/∂t = D_v ∇²V + r*V*(1 - V/K) - c*G*V + β*G*V*exp(-α*G²) - m*V
  ∂G/∂t = D_g ∇²G + e*c*G*V - d*G

Where facilitation term β*G*V*exp(-α*G²) captures:
- Nutrient cycling (dung/urine)
- Trampling (seed-soil contact)
"""

import numpy as np
from typing import Tuple, Optional
from .parameters import TibetanPlateauParams


class GrazerVegetationModel:
    """
    Spatial grazer-vegetation model with facilitation
    """

    def __init__(self, params: TibetanPlateauParams):
        """
        Initialize model with parameters

        Args:
            params: TibetanPlateauParams instance
        """
        self.p = params

        # Spatial operators (for finite differences)
        self._setup_laplacian()

    def _setup_laplacian(self):
        """
        Precompute Laplacian operator coefficients for efficiency

        Uses 5-point stencil for ∇²u:
        ∇²u ≈ (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / h²
        """
        self.laplacian_coeff = 1.0 / (self.p.dx * self.p.dy)

    def laplacian(self, u: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian using finite differences with periodic BCs

        Args:
            u: 2D array (nx, ny)

        Returns:
            ∇²u: 2D array (nx, ny)
        """
        # Periodic boundary conditions (wrapping)
        laplacian = (
            np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
            np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) -
            4 * u
        ) * self.laplacian_coeff

        return laplacian

    def vegetation_rhs(self, V: np.ndarray, G: np.ndarray) -> np.ndarray:
        """
        Right-hand side of vegetation equation

        dV/dt = D_v ∇²V + r*V*(1 - V/K) - c*G*V + β*G*V*exp(-α*G²) - m*V
                ^^^^^^^^   ^^^^^^^^^^^^^^   ^^^^^^   ^^^^^^^^^^^^^^^^^^   ^^^^
                diffusion  logistic growth  consumed    facilitation      mortality

        Args:
            V: Vegetation biomass (g/m²)
            G: Grazer density (sheep units/ha)

        Returns:
            dV/dt
        """
        # Diffusion
        diffusion = self.p.D_v * self.laplacian(V)

        # Growth (logistic)
        growth = self.p.r * V * (1 - V / self.p.K)

        # Consumption (negative)
        consumption = -self.p.c * G * V

        # Facilitation (positive, hump-shaped in G)
        facilitation = self.p.beta * G * V * np.exp(-self.p.alpha * G**2)

        # Background mortality (negative)
        mortality = -self.p.m * V

        return diffusion + growth + consumption + facilitation + mortality

    def grazer_rhs(self, V: np.ndarray, G: np.ndarray) -> np.ndarray:
        """
        Right-hand side of grazer equation

        dG/dt = D_g ∇²G + e*c*G*V - d*G
                ^^^^^^^^   ^^^^^^^^   ^^^^
                movement   reproduction  mortality

        Args:
            V: Vegetation biomass (g/m²)
            G: Grazer density (sheep units/ha)

        Returns:
            dG/dt
        """
        # Movement (diffusion)
        movement = self.p.D_g * self.laplacian(G)

        # Reproduction (from vegetation consumption)
        reproduction = self.p.e * self.p.c * G * V

        # Mortality
        mortality = -self.p.d * G

        return movement + reproduction + mortality

    def rhs(self, state: np.ndarray) -> np.ndarray:
        """
        Combined right-hand side for both equations

        Args:
            state: Flattened state [V, G] of shape (2 * nx * ny,)

        Returns:
            dstate/dt of same shape
        """
        # Reshape from flat vector to spatial grids
        n_cells = self.p.nx * self.p.ny
        V = state[:n_cells].reshape(self.p.nx, self.p.ny)
        G = state[n_cells:].reshape(self.p.nx, self.p.ny)

        # Compute derivatives
        dV_dt = self.vegetation_rhs(V, G)
        dG_dt = self.grazer_rhs(V, G)

        # Flatten back to vector
        return np.concatenate([dV_dt.flatten(), dG_dt.flatten()])

    def step(self, V: np.ndarray, G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single forward Euler time step

        Args:
            V: Current vegetation (nx, ny)
            G: Current grazers (nx, ny)

        Returns:
            (V_new, G_new): Updated states
        """
        dV_dt = self.vegetation_rhs(V, G)
        dG_dt = self.grazer_rhs(V, G)

        V_new = V + self.p.dt * dV_dt
        G_new = G + self.p.dt * dG_dt

        # Ensure non-negativity
        V_new = np.maximum(V_new, 0)
        G_new = np.maximum(G_new, 0)

        return V_new, G_new

    def simulate(
        self,
        V0: np.ndarray,
        G0: np.ndarray,
        save_every: int = 100,
        progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run full simulation

        Args:
            V0: Initial vegetation (nx, ny)
            G0: Initial grazers (nx, ny)
            save_every: Save state every N steps
            progress: Print progress updates

        Returns:
            (t, V_history, G_history):
                t: Time points
                V_history: Vegetation over time (n_saves, nx, ny)
                G_history: Grazers over time (n_saves, nx, ny)
        """
        n_steps = int(self.p.t_max / self.p.dt)
        n_saves = n_steps // save_every + 1

        # Initialize storage
        t_saves = np.zeros(n_saves)
        V_history = np.zeros((n_saves, self.p.nx, self.p.ny))
        G_history = np.zeros((n_saves, self.p.nx, self.p.ny))

        # Initial condition
        V = V0.copy()
        G = G0.copy()
        V_history[0] = V
        G_history[0] = G
        t_saves[0] = 0

        save_idx = 1

        # Time integration
        for step in range(1, n_steps + 1):
            V, G = self.step(V, G)

            if step % save_every == 0:
                t_saves[save_idx] = step * self.p.dt
                V_history[save_idx] = V
                G_history[save_idx] = G

                if progress and save_idx % 10 == 0:
                    print(f"t = {t_saves[save_idx]:.1f} / {self.p.t_max:.1f} years, "
                          f"<V> = {V.mean():.1f} g/m², <G> = {G.mean():.2f} su/ha")

                save_idx += 1

        return t_saves, V_history, G_history

    def equilibrium_solver(
        self,
        G_mean: float,
        tol: float = 1e-3,
        max_iter: int = 10000
    ) -> Tuple[float, float]:
        """
        Find spatial equilibrium for a given mean grazer density

        Assumes spatially homogeneous equilibrium (∇² = 0)

        Args:
            G_mean: Fixed grazer density (sheep units/ha)
            tol: Convergence tolerance
            max_iter: Maximum iterations

        Returns:
            (V_eq, G_eq): Equilibrium values
        """
        # Initial guess
        V = self.p.K / 2
        G = G_mean

        for i in range(max_iter):
            # Vegetation equilibrium (set dV/dt = 0, ignoring diffusion)
            # r*V*(1 - V/K) - c*G*V + β*G*V*exp(-α*G²) - m*V = 0
            # V * [r*(1 - V/K) - c*G + β*G*exp(-α*G²) - m] = 0

            # Non-trivial solution
            numerator = self.p.r + self.p.beta * G * np.exp(-self.p.alpha * G**2) - self.p.m
            denominator = self.p.r / self.p.K + self.p.c * G

            if denominator > 0:
                V_new = numerator / denominator
                V_new = max(0, min(V_new, self.p.K))
            else:
                V_new = 0

            # Grazer equilibrium (set dG/dt = 0)
            # e*c*G*V - d*G = 0
            # G * (e*c*V - d) = 0

            if V_new > 0:
                # Non-trivial solution requires e*c*V = d
                # But we're fixing G, so just check consistency
                pass

            if abs(V_new - V) < tol:
                return V_new, G

            V = V_new

        print(f"Warning: Equilibrium solver did not converge after {max_iter} iterations")
        return V, G


def uniform_initial_condition(
    params: TibetanPlateauParams,
    V_mean: float,
    G_mean: float,
    noise: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create uniform initial condition with small spatial noise

    Args:
        params: Model parameters
        V_mean: Mean vegetation biomass
        G_mean: Mean grazer density
        noise: Noise amplitude (fraction of mean)

    Returns:
        (V0, G0): Initial conditions
    """
    rng = np.random.RandomState(42)

    V0 = V_mean * (1 + noise * (2 * rng.rand(params.nx, params.ny) - 1))
    G0 = G_mean * (1 + noise * (2 * rng.rand(params.nx, params.ny) - 1))

    V0 = np.maximum(V0, 0)
    G0 = np.maximum(G0, 0)

    return V0, G0


def patchy_initial_condition(
    params: TibetanPlateauParams,
    V_mean: float,
    G_mean: float,
    n_patches: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create patchy initial condition (useful for pattern formation)

    Args:
        params: Model parameters
        V_mean: Mean vegetation biomass
        G_mean: Mean grazer density
        n_patches: Number of random patches

    Returns:
        (V0, G0): Initial conditions
    """
    rng = np.random.RandomState(42)

    V0 = np.ones((params.nx, params.ny)) * V_mean * 0.5
    G0 = np.ones((params.nx, params.ny)) * G_mean

    # Add random patches
    for _ in range(n_patches):
        cx = rng.randint(0, params.nx)
        cy = rng.randint(0, params.ny)
        r = params.L / 10

        for i in range(params.nx):
            for j in range(params.ny):
                dist = np.sqrt((params.x[i] - params.x[cx])**2 +
                             (params.y[j] - params.y[cy])**2)
                if dist < r:
                    V0[i, j] = V_mean * 1.5

    return V0, G0
