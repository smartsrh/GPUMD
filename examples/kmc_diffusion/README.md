# KMC Diffusion Example

## Files

- `kmc_params.txt`: KMC parameters (Ef, Em for each solute)
- `run.in`: GPUMD input file

## Usage

1. Prepare your structure (model.xyz) with Cu matrix and solute atoms
2. Adjust `kmc_params.txt` with your solute parameters
3. Run: `gpumd`

## Parameter File Format

```
# species  Ef(eV)  Em(eV)  [mass(amu)]
Zr         1.35    1.8     91.22
Cr         1.55    0.8     52.00
```

- **species**: Element symbol
- **Ef**: Vacancy formation energy (eV)
- **Em**: Migration energy (eV)
- **mass**: Atomic mass (optional)

## Command

```
kmc_diffusion <T(K)> <t_max(s)> [param_file] [dump_interval]
```

### Examples

```bash
# Use default kmc_params.txt
kmc_diffusion 800 1e-6

# Use custom parameter file
kmc_diffusion 800 1e-6 my_params.txt

# With dump interval
kmc_diffusion 800 1e-6 my_params.txt 500
```

## Output

- `kmc_diffusion.out`: Diffusion coefficients for each solute
