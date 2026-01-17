"""
Parameters for Tibetan Plateau Alpine Grassland Model

Based on literature values for:
- Alpine grassland ecosystems (3000-5000m elevation)
- Kobresia, Stipa-dominated communities
- Yak/sheep/goat grazing systems
- Short growing season (June-September, ~100 days)

Units:
- Vegetation V: g/m² (aboveground biomass)
- Grazers G: sheep units/ha (1 yak = 5 sheep units)
- Time: years
- Space: meters
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class TibetanPlateauParams:
    """Default parameters calibrated for Tibetan Plateau alpine grasslands"""

    # === VEGETATION PARAMETERS ===
    r: float = 0.4  # Growth rate (/year)
                    # Low due to short growing season and harsh conditions
                    # Literature: 0.3-0.5 /year

    K: float = 180.0  # Carrying capacity (g/m²)
                      # Typical alpine grassland peak biomass
                      # Literature: 150-200 g/m² for healthy grassland

    m: float = 0.12  # Background mortality (/year)
                     # Winter dieback, environmental stress
                     # Literature: 0.1-0.15 /year

    D_v: float = 0.5  # Vegetation dispersal (m²/day)
                      # Short-range seed dispersal
                      # Literature: 0.1-1 m²/day for alpine grasses

    # === GRAZING PARAMETERS ===
    c: float = 1.0  # Consumption rate (g/sheep-unit/day)
                    # Per capita consumption rate
                    # Literature: 0.5-2 g/sheep-unit/day

    e: float = 0.08  # Conversion efficiency (dimensionless)
                     # Low due to harsh environment, high maintenance costs
                     # Literature: 0.05-0.1 for harsh alpine environments

    d: float = 0.15  # Grazer mortality (/year)
                     # Natural mortality rate
                     # Literature: 0.1-0.2 /year

    D_g: float = 50.0  # Grazer movement (m²/day)
                       # Mobile herding, relatively high mobility
                       # Literature: 10-100 m²/day depending on herding strategy

    # === FACILITATION PARAMETERS (KEY FOR STORY) ===
    beta: float = 0.10  # Facilitation strength (dimensionless)
                        # Benefit from nutrient cycling + trampling
                        # THIS PARAMETER determines if grazing can be beneficial
                        # Range: 0.05-0.15
                        # Higher β = stronger beneficial effect

    alpha: float = 1.0  # Facilitation decay rate (ha/sheep-unit)
                        # Controls optimal grazing density
                        # Higher α = sharper peak, narrower beneficial range
                        # Range: 0.5-1.5

    # === SPATIAL DOMAIN ===
    L: float = 100.0  # Domain size (meters)
                      # Represents a grassland patch

    nx: int = 100  # Grid points in x
    ny: int = 100  # Grid points in y

    # === TEMPORAL ===
    t_max: float = 100.0  # Simulation time (years)
    dt: float = 0.01  # Time step (years) ≈ 3.65 days

    # === DERIVED PROPERTIES ===
    def __post_init__(self):
        """Calculate derived spatial properties"""
        self.dx = self.L / self.nx
        self.dy = self.L / self.ny
        self.x = np.linspace(0, self.L, self.nx)
        self.y = np.linspace(0, self.L, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    # === HELPER METHODS ===
    def facilitation(self, G):
        """
        Facilitation function: hump-shaped benefit from grazing

        f(G) = β * G * exp(-α * G²)

        Mechanisms:
        - Nutrient cycling: Dung/urine create nutrient hotspots
        - Trampling: Improves seed-soil contact

        Args:
            G: Grazer density (sheep units/ha)

        Returns:
            Facilitation effect (dimensionless)
        """
        return self.beta * G * np.exp(-self.alpha * G**2)

    def optimal_grazing_density(self):
        """
        Calculate grazing density that maximizes facilitation

        df/dG = 0 => G_opt = 1/√(2α)

        Returns:
            Optimal grazing density (sheep units/ha)
        """
        return 1.0 / np.sqrt(2 * self.alpha)

    def max_facilitation(self):
        """Maximum facilitation value at optimal grazing"""
        G_opt = self.optimal_grazing_density()
        return self.facilitation(G_opt)

    def __repr__(self):
        """Formatted parameter display"""
        G_opt = self.optimal_grazing_density()
        f_max = self.max_facilitation()

        return f"""
TibetanPlateauParams:
  Vegetation: r={self.r}, K={self.K}, m={self.m}, D_v={self.D_v}
  Grazing: c={self.c}, e={self.e}, d={self.d}, D_g={self.D_g}
  Facilitation: β={self.beta}, α={self.alpha}

  Optimal grazing density: {G_opt:.3f} sheep units/ha
  Maximum facilitation: {f_max:.4f}

  Domain: {self.L}m × {self.L}m, grid: {self.nx}×{self.ny}
  Time: {self.t_max} years, dt={self.dt} years
"""


# === PARAMETER VARIATIONS ===

def no_facilitation_params():
    """Parameters with no grazing benefit (control scenario)"""
    params = TibetanPlateauParams()
    params.beta = 0.0
    return params


def strong_facilitation_params():
    """Parameters with strong grazing benefit"""
    params = TibetanPlateauParams()
    params.beta = 0.15
    return params


def degraded_params():
    """Parameters for degraded grassland (lower productivity)"""
    params = TibetanPlateauParams()
    params.K = 120.0  # Lower carrying capacity
    params.r = 0.3    # Slower growth
    params.m = 0.18   # Higher mortality
    return params
