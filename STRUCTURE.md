# Grazer-Vegetation Model Structure

## Project Goal
Demonstrate that moderate grazing maintains alpine grassland vegetation on the Tibetan plateau, while over- or under-grazing leads to degradation/desertification.

## Model Structure

### Minimal Spatial Reaction-Diffusion System

**Vegetation dynamics:**
```
∂V/∂t = D_v ∇²V + r*V*(1 - V/K) - c*G*V + β*G*V*exp(-α*G²) - m*V
```

**Grazer dynamics:**
```
∂G/∂t = D_g ∇²G + e*c*G*V - d*G
```

### Parameters (Tibetan Plateau Calibration)

#### Vegetation Parameters
- `r`: Growth rate (0.3-0.5 /year) - limited by short growing season
- `K`: Carrying capacity (150-200 g/m²) - typical alpine grassland biomass
- `m`: Background mortality (0.1-0.15 /year) - winter dieback
- `D_v`: Dispersal coefficient (0.1-1 m²/day) - short-range seed dispersal

#### Grazing Parameters
- `c`: Consumption rate (0.5-2 g/sheep-unit/day)
- `β`: Facilitation strength (0.05-0.15) - **KEY PARAMETER**
- `α`: Facilitation decay (0.5-1.5) - determines optimal grazing range
- `e`: Conversion efficiency (0.05-0.1) - harsh environment
- `d`: Grazer mortality (0.1-0.2 /year)
- `D_g`: Grazer movement (10-100 m²/day) - mobile herding

### Key Mechanisms (Not Fire)
1. **Nutrient cycling**: Dung and urine create nutrient hotspots → enhanced growth
2. **Trampling**: Improves seed-soil contact → better germination
3. **Selective grazing**: Prevents dominance by unpalatable species (implicit in model)

### Implementation Plan

```
grazer/
├── README.md
├── STRUCTURE.md                    # This file
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── model.py                   # Core spatial PDE model
│   ├── parameters.py              # Tibetan plateau parameters
│   └── solver.py                  # PDE numerical integration
├── notebooks/
│   ├── 01_beneficial_range.ipynb  # Main story demonstration
│   └── 02_spatial_patterns.ipynb  # Desertification patterns
└── experiments/
    └── parameter_sweep.py          # Find optimal grazing range
```

## Story to Tell

### Three Regimes

1. **No/Low Grazing (G < G_opt)**:
   - Reduced nutrient cycling
   - Potential for senescent biomass accumulation
   - Lower vegetation productivity
   - Vulnerable to environmental shocks

2. **Optimal Grazing (G ≈ G_opt)**:
   - **Maximum facilitation benefit**
   - Stable, productive grassland
   - Spatial homogeneity maintained
   - **Desertification avoided**

3. **Overgrazing (G > G_opt)**:
   - Consumption overwhelms facilitation
   - Vegetation depletion
   - Spatial desertification patterns emerge
   - **System collapse**

### Visualizations

1. **Bifurcation diagram**: V_equilibrium vs. G
2. **Spatial snapshots**: Vegetation patterns at different grazing intensities
3. **Time series**: Three scenarios side-by-side
4. **Parameter space**: (β, α) heatmap showing beneficial region

## Literature Context

- Tibetan plateau: ~2.5 million km² alpine grassland
- Major issue: 90% degraded from overgrazing (since 1980s)
- Traditional nomadic grazing was sustainable (mobile, moderate density)
- Key species: Kobresia, Stipa, Carex
- Growing season: ~100 days (June-September)
- Livestock: Yaks, Tibetan sheep, goats (converted to "sheep units")
