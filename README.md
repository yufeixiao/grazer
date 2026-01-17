# Grazer-Vegetation Model: Beneficial Grazing in Tibetan Plateau Grasslands

A minimal spatial model demonstrating that **moderate grazing maintains vegetation and prevents desertification** in alpine grassland ecosystems.

## The Story

There exists a **parameter range in which grazers are beneficial** for vegetation maintenance, avoiding desertification. This model shows three distinct regimes:

1. **No/Low Grazing**: Reduced productivity, vulnerable system
2. **Optimal Grazing** ✓: Maximum vegetation health, desertification avoided
3. **Overgrazing** ✗: Vegetation depletion → desertification

## Model Overview

### Minimal Spatial Reaction-Diffusion System

**Vegetation dynamics:**
```
∂V/∂t = D_v ∇²V + r·V·(1 - V/K) - c·G·V + β·G·V·exp(-α·G²) - m·V
```

**Grazer dynamics:**
```
∂G/∂t = D_g ∇²G + e·c·G·V - d·G
```

### Key Mechanisms (Facilitation)

The hump-shaped term `β·G·V·exp(-α·G²)` represents grazing benefits:
- **Nutrient cycling**: Dung and urine create nutrient hotspots
- **Trampling**: Improves seed-soil contact for germination

### Calibration: Tibetan Plateau Alpine Grasslands

- Ecosystem: Alpine grassland (3000-5000m elevation)
- Species: Kobresia, Stipa-dominated communities
- Grazers: Yaks, sheep, goats (in "sheep units")
- Growing season: June-September (~100 days)
- Context: 90% degraded from overgrazing since 1980s

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Demo Notebook

```bash
jupyter notebook notebooks/01_beneficial_range.ipynb
```

This notebook demonstrates:
1. Facilitation function shape
2. **Bifurcation diagram** (core result showing beneficial range)
3. Time series comparison (3 scenarios)
4. Spatial patterns of desertification
5. Parameter sensitivity analysis

### Run Parameter Sweep

```bash
python experiments/parameter_sweep.py
```

Identifies the beneficial grazing range and generates parameter space heatmaps.

## Project Structure

```
grazer/
├── README.md                       # This file
├── STRUCTURE.md                    # Detailed model design
├── requirements.txt
├── src/
│   ├── model.py                   # Core spatial PDE model
│   ├── parameters.py              # Tibetan plateau calibration
│   └── solver.py                  # Numerical methods
├── notebooks/
│   └── 01_beneficial_range.ipynb  # Main demonstration
└── experiments/
    └── parameter_sweep.py          # Parameter space exploration
```

## Key Results

### Bifurcation Diagram

![Bifurcation](notebooks/bifurcation_diagram.png)

Shows vegetation equilibrium vs. grazing density. The **green curve above red** in the optimal range demonstrates that **grazing is beneficial** when facilitation mechanisms are present.

### Optimal Grazing Density

For default parameters:
- **Optimal density**: ~0.7 sheep units/ha
- **Beneficial range**: 0.35-1.05 sheep units/ha
- **Vegetation benefit**: Up to 15% higher than no grazing
- **Overgrazing threshold**: >1.5 sheep units/ha → collapse

## Management Implications

1. **Traditional nomadic grazing was sustainable**: Mobile, moderate density matched optimal range
2. **Current degradation drivers**: Sedentarization + overgrazing + climate change
3. **Complete grazing exclusion may not be optimal**: Moderate grazing enhances ecosystem function
4. **Target density**: ~0.7 sheep units/ha maintains grassland health

## References

Model based on empirical understanding of:
- Alpine grassland productivity (150-200 g/m² peak biomass)
- Short growing season dynamics (r = 0.3-0.5 /year)
- Grazing facilitation mechanisms (nutrient cycling, trampling)
- Tibetan plateau degradation patterns

## License

MIT License

## Contact

For questions about the model or collaboration: [Your contact info]
