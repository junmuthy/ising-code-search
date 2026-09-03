# Initial ideal TMR results

The implementation exactly certifies the underlying code as `[[9,1,3]]`:

- `rank(H_X) = rank(H_Z) = 4`;
- all CSS checks commute;
- the selected `X_L` and `Z_L` commute with the checks and anticommute with
  each other;
- exhaustive enumeration gives `d_X = d_Z = 3`;
- the synthesized Clifford encoder gives expectation `+1` for all eight code
  stabilizers and for `X_L`.

The runs used Python 3.12.3, `bloqade-tsim==0.1.5`, JAX 0.11.1, and the CPU
backend.

## Direct projection circuit

Each entry below used 20,000 shots per logical observable. The three observable
circuits have the same state preparation and projection but independent samples.

| Target angle | Quantity | Sampled | Ideal |
| ---: | :--- | ---: | ---: |
| `0` | acceptance | `1.0000` | `1.0000` |
| `0` | `<X_L>` | `1.0000` | `1.0000` |
| `0` | `<Y_L>` | `-0.0009` | `0.0000` |
| `0` | `<Z_L>` | `-0.0036` | `0.0000` |
| `0.01` | acceptance | `0.9168`–`0.9194` | `0.9172` |
| `0.01` | `<X_L>` | `1.0000` | `0.99995` |
| `0.01` | `<Y_L>` | `0.0112` | `0.0100` |
| `0.01` | `<Z_L>` | `0.0136` | `0.0000` |
| `pi/8` | acceptance | `0.4292`–`0.4345` | `0.4313` |
| `pi/8` | `<X_L>` | `0.9274` | `0.9239` |
| `pi/8` | `<Y_L>` | `0.3922` | `0.3827` |
| `pi/8` | `<Z_L>` | `0.0305` | `0.0000` |

All residuals are compatible with the recorded binomial sampling errors. The
`pi/8` point clearly resolves the positive logical-Y component and therefore
checks the otherwise subtle sign of the three-part physical rotation.

## Explicit syndrome-extraction circuit

The scheduled circuit uses 9 data qubits, 8 syndrome ancillas, four CNOT layers
per extraction round, two initialization rounds, and two post-TMR rounds. Because
canonical Stim physical IDs are retained, `tsim` reports `num_qubits=26`; only
the 17 listed physical IDs are active. It has 32 stabilizer detectors and one
terminal logical observable.

At target angle `0.01`, 5,000 shots per observable gave:

| Quantity | Sampled | Ideal |
| :--- | ---: | ---: |
| acceptance | `0.9094`–`0.9170` | `0.9172` |
| `<X_L>` | `1.0000` | `0.99995` |
| `<Y_L>` | `-0.0002 ± 0.0148` | `0.0100` |
| `<Z_L>` | `-0.0055 ± 0.0148` | `0.0000` |

The small logical-Y signal is below the resolution of this intentionally short
run. A second scheduled run at `theta=pi/8`, again with 5,000 shots per
observable, resolves the Bloch-vector rotation:

| Quantity | Sampled | Ideal |
| :--- | ---: | ---: |
| acceptance | `0.4230`–`0.4338` | `0.4313` |
| `<X_L>` | `0.9281 ± 0.0080` | `0.9239` |
| `<Y_L>` | `0.3636 ± 0.0203` | `0.3827` |
| `<Z_L>` | `-0.0185 ± 0.0215` | `0.0000` |

The explicit ancilla circuit therefore implements the same syndrome projection
and accepted logical rotation as the ideal MPP presentation.

## Scope

These are noiseless mechanism checks, not logical-fidelity estimates. In
particular, they do not yet include measurement-based encoded-state
initialization, physical gate noise, decoding, optimized postselection, an
inverse-reference patch, or RUS teleportation. The raw JSON files contain exact
shot counts, detector histograms, independent acceptance estimates, standard
errors, compilation times, and sampling times.
