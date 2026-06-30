(targets)=

# Target Functions

Target functions define what properties you're optimizing against. DESCENT provides several built-in target types and allows for custom implementations.

## Overview

A target function computes a loss by comparing force field predictions to reference data. The overall training loss is typically a weighted sum of multiple target losses:

$$\mathcal{L}_{\text{total}} = \sum_i w_i \mathcal{L}_i$$

where $w_i$ is the weight for target $i$.

## Energy Targets

Energy targets fit to reference energies (typically from quantum mechanics):

```python
from descent.targets.energy import EnergyTarget

target = EnergyTarget(
    dataset="qm_energies.json",
    weight=1.0,
    denominator="std",  # Normalize by standard deviation
    energy_type="absolute"  # or "relative"
)
```

### Energy Dataset Format

Energy datasets should contain:
- Molecular structures (SMILES or conformers)
- Reference energies
- Optional: weights per structure

```json
{
  "entries": [
    {
      "smiles": "CCO",
      "conformers": [...],
      "energies": [0.0, 1.2, 2.3],
      "weights": [1.0, 1.0, 0.5]
    }
  ]
}
```

### Loss Computation

For absolute energies:

$$\mathcal{L}_{\text{energy}} = \frac{1}{N} \sum_{i=1}^{N} \frac{(E_i^{\text{pred}} - E_i^{\text{ref}})^2}{\sigma^2}$$

For relative energies (differences from minimum):

$$\mathcal{L}_{\text{energy}} = \frac{1}{N} \sum_{i=1}^{N} \frac{(\Delta E_i^{\text{pred}} - \Delta E_i^{\text{ref}})^2}{\sigma^2}$$

## Thermodynamic Targets

Thermodynamic targets fit to experimental or simulated properties:

```python
from descent.targets.thermo import ThermoTarget

target = ThermoTarget(
    dataset="thermo.json",
    weight=0.5,
    properties=["density", "hvap", "dielectric"],
    temperatures=[298.15],  # K
    pressures=[1.0]  # atm
)
```

### Supported Properties

- **Density** ($\rho$): Mass density in g/cm³
- **Heat of vaporization** ($\Delta H_{\text{vap}}$): In kJ/mol
- **Dielectric constant** ($\epsilon$): Dimensionless
- **Heat capacity** ($C_p$): In J/(mol·K)
- **Thermal expansion** ($\alpha$): In K⁻¹
- **Compressibility** ($\kappa$): In atm⁻¹

### Thermodynamic Dataset Format

```json
{
  "entries": [
    {
      "smiles": "CCO",
      "properties": {
        "density": {"value": 0.789, "uncertainty": 0.001},
        "hvap": {"value": 42.3, "uncertainty": 0.5}
      },
      "temperature": 298.15,
      "pressure": 1.0
    }
  ]
}
```

### Loss Computation

$$\mathcal{L}_{\text{thermo}} = \sum_{j} w_j \frac{1}{N_j} \sum_{i=1}^{N_j} \left(\frac{p_{ij}^{\text{pred}} - p_{ij}^{\text{ref}}}{\sigma_{ij}}\right)^2$$

where $j$ indexes properties, $i$ indexes molecules, and $\sigma_{ij}$ is the experimental uncertainty.

## Dimer Targets

Dimer targets fit to non-bonded interaction energies:

```python
from descent.targets.dimers import DimerTarget

target = DimerTarget(
    dataset="dimers.json",
    weight=0.3,
    distance_range=(2.5, 10.0),  # Angstroms
)
```

### Dimer Dataset Format

```json
{
  "entries": [
    {
      "smiles_a": "O",
      "smiles_b": "O",
      "geometries": [...],
      "distances": [2.5, 3.0, 3.5, 4.0],
      "energies": [-5.2, -4.1, -2.3, -0.8]
    }
  ]
}
```

### Loss Computation

$$\mathcal{L}_{\text{dimer}} = \frac{1}{N} \sum_{i=1}^{N} \frac{(E_i^{\text{int,pred}} - E_i^{\text{int,ref}})^2}{\sigma^2}$$

where $E_i^{\text{int}} = E_{AB} - E_A - E_B$ is the interaction energy.

## Custom Targets

Create custom target functions by subclassing `BaseTarget`:

```python
from descent.targets import BaseTarget
import torch

class CustomTarget(BaseTarget):
    def __init__(self, dataset, weight=1.0, **kwargs):
        super().__init__(weight=weight)
        self.dataset = self.load_dataset(dataset)
        
    def compute_loss(self, system, parameters):
        """
        Compute loss for this target.
        
        Parameters
        ----------
        system : System
            The molecular system
        parameters : dict
            Current force field parameters
            
        Returns
        -------
        loss : torch.Tensor
            Scalar loss value
        """
        predictions = self.predict(system, parameters)
        references = self.get_references()
        
        loss = torch.mean((predictions - references) ** 2)
        return loss
        
    def predict(self, system, parameters):
        """Compute predictions using current parameters."""
        # Your implementation here
        pass
        
    def get_references(self):
        """Get reference values."""
        return torch.tensor(self.dataset['values'])
```

## Combining Targets

Multiple targets are combined in the training configuration:

```python
from descent.train import TrainingConfig

config = TrainingConfig(
    force_field="openff-2.0.0.offxml",
    targets=[
        {"type": "energy", "dataset": "qm.json", "weight": 1.0},
        {"type": "thermo", "dataset": "exp.json", "weight": 0.5},
        {"type": "dimer", "dataset": "dimers.json", "weight": 0.3},
    ]
)
```

## Weight Selection

Choosing appropriate weights is critical:

1. **Scale normalization**: Normalize each target to similar scale (0-1 or 0-10)
2. **Relative importance**: Weight by physical importance or data quality
3. **Data size**: Consider number of data points per target
4. **Units**: Ensure consistent units across targets

Example weight selection:
```python
# Option 1: Equal contribution
weights = {"energy": 1.0, "thermo": 1.0, "dimer": 1.0}

# Option 2: Prioritize QM data
weights = {"energy": 2.0, "thermo": 1.0, "dimer": 0.5}

# Option 3: Scale by uncertainty
weights = {
    "energy": 1.0 / energy_std**2,
    "thermo": 1.0 / thermo_std**2,
}
```

## See Also

- [Training](training.md): Training workflows
- [Optimization](optimization.md): Optimization strategies
- [API Reference](api/targets.rst): Complete API documentation
