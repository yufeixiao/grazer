"""
Parameter sweep to identify beneficial grazing range

Explores how facilitation strength (β) and grazing density (G)
affect vegetation equilibrium.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from src.model import GrazerVegetationModel
from src.parameters import TibetanPlateauParams


def sweep_grazing_density(beta_range, G_range, save_plot=True):
    """
    2D parameter sweep: (β, G) -> V_equilibrium

    Args:
        beta_range: Array of facilitation strengths
        G_range: Array of grazing densities
        save_plot: Save figure to file

    Returns:
        V_eq: 2D array of equilibrium vegetation (len(beta), len(G))
    """
    V_eq = np.zeros((len(beta_range), len(G_range)))

    for i, beta in enumerate(beta_range):
        print(f"β = {beta:.3f}")
        params = TibetanPlateauParams()
        params.beta = beta
        model = GrazerVegetationModel(params)

        for j, G in enumerate(G_range):
            V, _ = model.equilibrium_solver(G)
            V_eq[i, j] = V

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.contourf(G_range, beta_range, V_eq, levels=20, cmap='YlGn')
    plt.colorbar(im, ax=ax, label='Equilibrium Vegetation (g/m²)')

    # Add contour lines
    contours = ax.contour(G_range, beta_range, V_eq, levels=10,
                          colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8)

    # Mark desertification threshold
    params_ref = TibetanPlateauParams()
    desert_threshold = params_ref.K * 0.3
    ax.contour(G_range, beta_range, V_eq, levels=[desert_threshold],
              colors='red', linewidths=2, linestyles='--')

    ax.set_xlabel('Grazing Density (sheep units/ha)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Facilitation Strength β', fontsize=13, fontweight='bold')
    ax.set_title('Vegetation Equilibrium: Beneficial Grazing Parameter Space',
                fontsize=15, fontweight='bold')

    # Add text annotation
    ax.text(0.98, 0.98, 'Red dashed line:\nDesertification threshold',
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_plot:
        plt.savefig('../notebooks/parameter_space_heatmap.png', dpi=150, bbox_inches='tight')
        print("\nFigure saved: parameter_space_heatmap.png")

    plt.show()

    return V_eq


def find_optimal_range(beta=0.10, threshold=0.95):
    """
    Find grazing range where facilitation provides benefit

    Args:
        beta: Facilitation strength
        threshold: Fraction of maximum vegetation to define "beneficial"

    Returns:
        (G_min, G_max): Beneficial grazing range
    """
    params = TibetanPlateauParams()
    params.beta = beta
    model = GrazerVegetationModel(params)

    # Get equilibrium curve
    G_range = np.linspace(0, 3, 500)
    V_eq = []

    for G in G_range:
        V, _ = model.equilibrium_solver(G)
        V_eq.append(V)

    V_eq = np.array(V_eq)

    # Reference: no grazing
    V_no_grazing = V_eq[0]

    # Find where V > threshold * V_no_grazing
    beneficial = V_eq > threshold * V_no_grazing

    if beneficial.any():
        beneficial_indices = np.where(beneficial)[0]
        G_min = G_range[beneficial_indices[0]]
        G_max = G_range[beneficial_indices[-1]]

        print(f"\n{'='*60}")
        print(f"BENEFICIAL GRAZING RANGE (β = {beta:.2f})")
        print(f"{'='*60}")
        print(f"Range: {G_min:.3f} - {G_max:.3f} sheep units/ha")
        print(f"Width: {G_max - G_min:.3f} sheep units/ha")
        print(f"\nOptimal density: {params.optimal_grazing_density():.3f} sheep units/ha")
        print(f"Max vegetation in range: {V_eq[beneficial].max():.1f} g/m²")
        print(f"Vegetation without grazing: {V_no_grazing:.1f} g/m²")
        print(f"Benefit: {(V_eq[beneficial].max() / V_no_grazing - 1)*100:.1f}% increase")
        print(f"{'='*60}\n")

        return G_min, G_max
    else:
        print(f"\nNo beneficial range found for β = {beta:.2f}")
        return None, None


if __name__ == "__main__":
    print("="*60)
    print("BENEFICIAL GRAZING RANGE ANALYSIS")
    print("Tibetan Plateau Alpine Grasslands")
    print("="*60)

    # 1. Find optimal range for default parameters
    print("\n1. Finding beneficial range with default facilitation...")
    find_optimal_range(beta=0.10)

    # 2. Test different facilitation strengths
    print("\n2. Testing sensitivity to facilitation strength...")
    for beta in [0.05, 0.10, 0.15, 0.20]:
        find_optimal_range(beta=beta)

    # 3. Generate 2D parameter space map
    print("\n3. Generating parameter space heatmap...")
    beta_range = np.linspace(0, 0.25, 50)
    G_range = np.linspace(0, 2.5, 100)

    V_eq_map = sweep_grazing_density(beta_range, G_range, save_plot=True)

    print("\n✅ Analysis complete!")
    print("📊 Results saved to notebooks/parameter_space_heatmap.png")
