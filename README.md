# 🌊 Hydrogen Diffusion Simulation
### *Mathematical Modeling of H₂ Transport in PEM Membranes*

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Powered-green.svg)
![SciPy](https://img.shields.io/badge/SciPy-Scientific-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

*A computational framework for simulating hydrogen purification and storage in PEM electrolysers using finite difference methods*

[Features](#-key-features) • [Installation](#-installation) • [Usage](#-usage) • [Theory](#-mathematical-foundation) • [Results](#-results) • [Contributing](#-contributing)

</div>

---

## 🎯 Overview

This repository implements **state-of-the-art numerical simulations** for hydrogen diffusion through Proton Exchange Membrane (PEM) materials, a critical component in clean energy systems. As the world transitions toward sustainable energy, understanding hydrogen transport at the microscale is essential for optimizing electrolyser design and hydrogen storage systems.

Our simulation toolkit employs **Finite Difference Methods (FDM)** to solve both steady-state and transient diffusion problems, providing insights into:
- Concentration profile evolution through membrane thickness
- Hydrogen flux optimization under varying operating conditions
- Effects of membrane microstructure (porosity and tortuosity)
- Temperature and pressure dependencies on transport properties

### 🌟 Why This Matters

Hydrogen is projected to play a pivotal role in decarbonization, with PEM electrolysers being a key technology for green hydrogen production. This simulation framework enables:
- **Design Optimization**: Predict membrane performance before costly experimental trials
- **Process Understanding**: Visualize molecular transport phenomena in real-time
- **Parameter Studies**: Systematically explore operating condition effects
- **Educational Tool**: Learn numerical methods applied to real engineering problems

---

## 🚀 Key Features

### ✨ Simulation Capabilities

| Feature | Description |
|---------|-------------|
| **Dual-State Analysis** | Solve both steady-state and transient diffusion problems |
| **Multiple Solvers** | Linear algebra (`scipy.linalg.solve`) and nonlinear root finding (`scipy.optimize`) |
| **Physical Realism** | Temperature-dependent permeability with Arrhenius kinetics |
| **Microstructure Effects** | Incorporate porosity (ε) and tortuosity (τ) parameters |
| **Parametric Studies** | Systematic variation of thickness, pressure, temperature |
| **Visualization Suite** | Publication-quality plots of concentration profiles and flux maps |

### 🔬 Governed Physics

Our simulations are based on **Fick's Laws of Diffusion**:

**Steady State:**
```
d²C/dx² = 0
```

**Transient State:**
```
∂C/∂t = D ∂²C/∂x²
```

where `C` is hydrogen concentration, `D` is the diffusion coefficient, and `t` is time.

---

## 📋 Prerequisites

- Python 3.8 or higher
- NumPy (numerical operations)
- SciPy (scientific computing)
- Matplotlib (visualization)
- Jupyter Notebook (for interactive exploration)

---

## 🔧 Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/KevinChengKaiLai/Hydrogen-Diffusion-Simulation.git
cd Hydrogen-Diffusion-Simulation

# Create a virtual environment (recommended)
python -m venv hydrogen_env
source hydrogen_env/bin/activate  # On Windows: hydrogen_env\Scripts\activate

# Install dependencies
pip install numpy scipy matplotlib jupyter
```

### Verify Installation

```python
import numpy as np
import scipy
import matplotlib.pyplot as plt

print("All dependencies installed successfully!")
```

---

## 📖 Usage

### Steady-State Simulation

The steady-state simulation solves the diffusion equation at equilibrium, providing the concentration profile when temporal variations have ceased.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Physical parameters
L = 125e-6  # Membrane thickness (m)
D = 1e-10   # Diffusion coefficient (m²/s)
C_feed = 550.0  # Feed concentration
C_permeate = 316.0  # Permeate concentration

# Discretization
N = 100
x = np.linspace(0, L, N)
dx = L / (N - 1)

# Construct tridiagonal matrix A and vector b
A = np.zeros((N, N))
b = np.zeros(N)

# Boundary conditions
A[0, 0] = 1; b[0] = C_feed
A[-1, -1] = 1; b[-1] = C_permeate

# Interior points: finite difference approximation
for i in range(1, N-1):
    A[i, i-1] = -1
    A[i, i] = 2
    A[i, i+1] = -1

# Solve the linear system
C = solve(A, b)

# Visualize
plt.plot(x * 1e6, C)
plt.xlabel('Position (μm)')
plt.ylabel('Concentration (mol/m³)')
plt.title('Steady-State H₂ Concentration Profile')
plt.grid(True)
plt.show()
```

### Transient Simulation

The transient simulation captures time-dependent concentration evolution using the Method of Lines.

```python
from scipy.integrate import solve_ivp

# Physical parameters
L = 0.75e-6  # Membrane thickness (m)
D = 1e-8     # Diffusion coefficient (m²/s)

# Spatial discretization
N = 51
x = np.linspace(0, L, N)
dx = L / (N - 1)

# Boundary conditions
C_left = 1.0
C_right = 0.0

# Initial condition
C0 = np.zeros(N)
C0[0] = C_left
C0[-1] = C_right

# Define ODE system
def diffusion_ode(t, C):
    dCdt = np.zeros(N)
    for i in range(1, N-1):
        dCdt[i] = D * (C[i+1] - 2*C[i] + C[i-1]) / dx**2
    return dCdt

# Time integration
t_span = (0, 600)  # 10 minutes
t_eval = np.linspace(0, 600, 100)

sol = solve_ivp(diffusion_ode, t_span, C0, t_eval=t_eval, method='RK45')

# Animate or plot results
plt.figure(figsize=(10, 6))
for i in range(0, len(sol.t), 20):
    plt.plot(x * 1e6, sol.y[:, i], label=f't={sol.t[i]:.1f}s')
plt.xlabel('Position (μm)')
plt.ylabel('Concentration')
plt.title('Transient H₂ Diffusion')
plt.legend()
plt.grid(True)
plt.show()
```

### Running Jupyter Notebooks

Launch the interactive notebooks for exploratory analysis:

```bash
jupyter notebook
```

Then open:
- `project.ipynb` - Steady-state simulations and parametric studies
- `Transient.ipynb` - Time-dependent diffusion analysis
- `transient_0411.ipynb` - Advanced transient simulations

---

## 📐 Mathematical Foundation

### Governing Equations

#### Steady-State Diffusion

At equilibrium, the time derivative vanishes:

```
∇·(D∇C) = 0
```

For 1D with constant D:

```
d²C/dx² = 0
```

**Analytical Solution:**
```
C(x) = C₀ + (C_L - C₀) · (x/L)
```

#### Transient Diffusion

Time-dependent concentration evolution:

```
∂C/∂t = D · ∂²C/∂x²
```

#### Hydrogen Flux

From Fick's First Law:

```
J = -D · dC/dx
```

For membrane transport with pressure boundary conditions:

```
J = (P/L) · (√P_feed - √P_permeate)
```

where permeability `P` exhibits Arrhenius temperature dependence:

```
P = P₀ · exp(-E_a / RT)
```

### Numerical Discretization

#### Finite Difference Approximation

Second-order spatial derivative:

```
∂²C/∂x² ≈ (C_{i+1} - 2C_i + C_{i-1}) / Δx²
```

#### Matrix Form (Steady-State)

The discretized system becomes:

```
A · C = b
```

where A is a tridiagonal matrix:

```
A = [  1    0    0  ...  0  ]
    [ -1    2   -1  ...  0  ]
    [  0   -1    2  ...  0  ]
    [ ...            ...    ]
    [  0    0    0  ...  1  ]
```

#### Time Integration (Transient)

Using the Method of Lines, we convert the PDE into a system of ODEs:

```
dC/dt = (D/Δx²) · (C_{i+1} - 2C_i + C_{i-1})
```

Solved using `scipy.integrate.solve_ivp` with adaptive time-stepping (RK45 or BDF for stiff systems).

---

## 📊 Results

### Steady-State Analysis

Our simulations reveal several key findings:

#### 1. **Concentration Profiles**
Linear concentration gradients establish across the membrane thickness, consistent with Fick's Law predictions. The slope directly correlates with hydrogen flux.

#### 2. **Membrane Thickness Effects**
- **Thinner membranes** (5-20 μm): Higher flux, faster hydrogen recovery
- **Thicker membranes** (100-300 μm): Lower flux, improved mechanical stability
- **Optimal thickness**: ~20-50 μm balances flux and durability

#### 3. **Temperature Dependence**
Higher operating temperatures (450-500 K) significantly increase permeability and flux due to enhanced molecular mobility and activated diffusion.

#### 4. **Pressure Effects**
Hydrogen flux increases approximately linearly with feed pressure difference (ΔP = P_feed - P_permeate), validating Sieverts' Law behavior in metallic membranes.

#### 5. **Microstructure Impact**
- **Porosity (ε)**: Higher porosity (0.5-0.7) increases effective diffusivity
- **Tortuosity (τ)**: Higher tortuosity (2.0-2.5) reduces transport efficiency
- Optimal microstructure: ε=0.6, τ=1.5 maximizes flux while maintaining structure

### Transient Dynamics

#### 1. **Time to Steady State**
For typical membrane parameters (L=0.75 μm, D=1×10⁻⁸ m²/s), steady state is achieved in approximately 50-100 microseconds.

#### 2. **Concentration Wave Propagation**
The diffusion front propagates from the feed side, gradually establishing the equilibrium linear profile.

#### 3. **Response to Step Changes**
The membrane responds to feed pressure changes with characteristic relaxation times dependent on (L²/D).

---

## 🎨 Visualization Examples

The repository generates publication-quality figures including:

- ✅ Steady-state concentration profiles
- ✅ Hydrogen flux vs. membrane thickness
- ✅ Flux vs. temperature at various feed pressures
- ✅ Porosity and tortuosity parametric studies
- ✅ Transient concentration evolution snapshots
- ✅ Time-series concentration at specific spatial points

All figures are generated using Matplotlib with customizable aesthetics.

---

## 🗂️ Repository Structure

```
Hydrogen-Diffusion-Simulation/
│
├── README.md                   # This file
├── Report.pdf                  # Comprehensive technical report
│
├── project.ipynb              # Steady-state simulations
├── Transient.ipynb            # Basic transient analysis
├── trainsient_state.ipynb     # Extended transient studies
├── transient_0411.ipynb       # Latest transient implementations
│
├── requirements.txt           # Python dependencies
└── LICENSE                    # MIT License
```

---

## 🔬 Technical Details

### System Parameters

| Parameter | Symbol | Typical Range | Units |
|-----------|--------|---------------|-------|
| Membrane Thickness | L | 5 - 300 | μm |
| Diffusion Coefficient | D | 10⁻¹⁰ - 10⁻⁵ | m²/s |
| Temperature | T | 350 - 500 | K |
| Feed Pressure | P_feed | 100 - 400 | kPa |
| Permeate Pressure | P_perm | 100 | kPa |
| Porosity | ε | 0.3 - 0.7 | - |
| Tortuosity | τ | 1.5 - 2.5 | - |

### Computational Performance

- **Steady-state**: < 0.1 seconds (N=100 grid points)
- **Transient**: ~1-5 seconds (N=51, 100 time steps)
- **Memory**: Minimal (<50 MB for typical simulations)

---

## 🎓 Educational Value

This repository serves as an excellent resource for:

### Students
- Learn finite difference methods for PDEs
- Understand physical transport phenomena
- Practice scientific Python programming
- Develop numerical analysis skills

### Researchers
- Benchmark new hydrogen membrane materials
- Validate experimental measurements
- Design optimal membrane configurations
- Explore multi-physics coupling

### Engineers
- Predict electrolyser performance
- Optimize operating conditions
- Screen membrane candidates
- Support scale-up decisions

---

## 🚦 Future Roadmap

### Planned Enhancements

- [ ] **Multi-component diffusion** (H₂ + H₂O vapor)
- [ ] **2D/3D geometry** support for realistic membrane structures
- [ ] **Concentration-dependent diffusivity** (D = D(C))
- [ ] **Coupled heat and mass transfer**
- [ ] **Membrane degradation models**
- [ ] **GPU acceleration** for large-scale simulations
- [ ] **Interactive web dashboard** using Plotly/Dash
- [ ] **Experimental validation** with literature data
- [ ] **Machine learning surrogate models** for real-time prediction

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

1. **Report Bugs**: Open an issue describing the problem
2. **Suggest Features**: Propose new capabilities or improvements
3. **Submit Pull Requests**: Fix bugs or add features
4. **Improve Documentation**: Enhance README, add tutorials
5. **Share Results**: Publish your simulation findings

### Development Guidelines

```bash
# Fork the repository
# Create a feature branch
git checkout -b feature/amazing-feature

# Make your changes
# Commit with descriptive messages
git commit -m "Add: temperature-dependent porosity model"

# Push to your fork
git push origin feature/amazing-feature

# Open a Pull Request
```

---

## 📚 References

This work builds upon established theory and methods in membrane transport:

1. **Crank, J.** (1975). *The Mathematics of Diffusion*. Oxford University Press.
2. **Satterfield, C.N.** (1970). *Mass Transfer in Heterogeneous Catalysis*. MIT Press.
3. **Bird, R.B., Stewart, W.E., Lightfoot, E.N.** (2007). *Transport Phenomena*. Wiley.
4. **Carmo, M., et al.** (2013). "A comprehensive review on PEM water electrolysis." *Int. J. Hydrogen Energy*, 38(12), 4901-4934.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Cheng Kai Lai, Tejas Jairange, Neil Sinha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 👥 Authors

**Carnegie Mellon University - Mathematical Modelling in Chemical Engineering**

- **Cheng Kai Lai** - *Lead Developer*
- **Tejas Jairange** - *Transient Analysis*
- **Neil Sinha** - *Validation & Testing*

**Course**: 06-623 Mathematical Modelling in Chemical Engineering  
**Institution**: Carnegie Mellon University  
**Semester**: Spring 2025

---

## 💬 Contact

Have questions or suggestions? Reach out:

- 📧 Email: [Contact via GitHub](https://github.com/KevinChengKaiLai)
- 🐛 Issues: [GitHub Issues](https://github.com/KevinChengKaiLai/Hydrogen-Diffusion-Simulation/issues)
- 💡 Discussions: [GitHub Discussions](https://github.com/KevinChengKaiLai/Hydrogen-Diffusion-Simulation/discussions)

---

## 🌟 Acknowledgments

Special thanks to:
- Carnegie Mellon University Department of Chemical Engineering
- Course instructors for guidance on numerical methods
- The open-source Python scientific computing community
- All contributors and users of this simulation framework

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! It helps others discover this work.

[![Star History Chart](https://api.star-history.com/svg?repos=KevinChengKaiLai/Hydrogen-Diffusion-Simulation&type=Date)](https://star-history.com/#KevinChengKaiLai/Hydrogen-Diffusion-Simulation&Date)

---

<div align="center">

**Made with ❤️ for the hydrogen energy community**

*Advancing clean energy through computational science*

[⬆ Back to Top](#-hydrogen-diffusion-simulation)

</div>
