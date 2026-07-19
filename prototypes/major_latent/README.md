# Major–Latent Matrix Prototypes

Two small non-traffic examples of

\[
x=Wz,\qquad \widetilde B=BW.
\]

## Examples

1. **Cutting stock:** original columns are cutting patterns.
2. **Unit commitment:** original columns are feasible generator schedules.

Each compares:

- **H0:** full explicit column pool;
- **C:** fully compressed latent atoms;
- **RC:** explicit major columns plus latent minor atoms.

## Run

```bash
python -m pip install numpy scipy
python examples/run_all.py
```

These are LP-relaxation prototypes for testing the representation layer. They deliberately exclude branch-and-price, integer recovery, software integration, and transportation assignment.
