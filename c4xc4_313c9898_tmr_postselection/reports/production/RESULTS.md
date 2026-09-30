# 313c9898 one-/four-resource post-selection results

Both policies use identical trajectories at each point. Intervals are pointwise 95% Wilson intervals.
Accepted resources are candidates; conditional noisy logical fidelity has not been measured.

| Schedule | N | Angle | Strict XZ acceptance (95% CI) | X-only acceptance (95% CI) | Ideal |
|---|---:|---|---:|---:|---:|
| `original` | 1 | `0p001` | 16.743% [16.670, 16.816] | 27.221% [27.134, 27.309] | 98.134% |
| `original` | 1 | `pi128` | 14.753% [14.684, 14.823] | 23.891% [23.807, 23.975] | 85.610% |
| `original` | 1 | `pi32` | 11.843% [11.780, 11.907] | 19.178% [19.101, 19.255] | 68.715% |
| `original` | 1 | `zero` | 17.151% [17.077, 17.225] | 27.820% [27.732, 27.908] | 100.000% |
| `original` | 4 | `0p001` | 16.018% [15.946, 16.090] | 25.908% [25.823, 25.994] | 92.741% |
| `original` | 4 | `pi128` | 9.287% [9.230, 9.344] | 15.008% [14.939, 15.079] | 53.716% |
| `original` | 4 | `pi32` | 3.858% [3.820, 3.896] | 6.229% [6.182, 6.277] | 22.295% |
| `original` | 4 | `zero` | 17.324% [17.250, 17.399] | 27.995% [27.907, 28.083] | 100.000% |
| `preferred` | 1 | `0p001` | 17.568% [17.494, 17.643] | 29.254% [29.165, 29.344] | 98.134% |
| `preferred` | 1 | `pi128` | 15.387% [15.317, 15.458] | 25.581% [25.495, 25.667] | 85.610% |
| `preferred` | 1 | `pi32` | 12.302% [12.238, 12.367] | 20.471% [20.392, 20.550] | 68.715% |
| `preferred` | 1 | `zero` | 17.933% [17.858, 18.008] | 29.876% [29.787, 29.966] | 100.000% |
| `preferred` | 4 | `0p001` | 16.813% [16.740, 16.887] | 27.806% [27.719, 27.894] | 92.741% |
| `preferred` | 4 | `pi128` | 9.688% [9.631, 9.747] | 16.038% [15.966, 16.110] | 53.716% |
| `preferred` | 4 | `pi32` | 4.051% [4.013, 4.090] | 6.685% [6.637, 6.735] | 22.295% |
| `preferred` | 4 | `zero` | 18.138% [18.063, 18.214] | 29.899% [29.809, 29.989] | 100.000% |

## Cost at the pi/32 benchmark

| Schedule | N | Policy | Resources/attempt | Attempts/block | CNOTs/resource | Allocated qubit-CNOT-layers/resource |
|---|---:|---|---:|---:|---:|---:|
| `original` | 1 | `strict-xz` | 0.11843 | 8.444 | 9777.8 | 28100.8 |
| `original` | 1 | `tmr-x` | 0.19178 | 5.214 | 6038.2 | 17353.3 |
| `original` | 4 | `strict-xz` | 0.15432 | 25.921 | 7620.7 | 21566.1 |
| `original` | 4 | `tmr-x` | 0.24917 | 16.053 | 4719.7 | 13356.5 |
| `preferred` | 1 | `strict-xz` | 0.12302 | 8.129 | 7072.0 | 23670.9 |
| `preferred` | 1 | `tmr-x` | 0.20471 | 4.885 | 4250.0 | 14225.1 |
| `preferred` | 4 | `strict-xz` | 0.16205 | 24.684 | 5479.9 | 17970.0 |
| `preferred` | 4 | `tmr-x` | 0.26742 | 14.958 | 3320.6 | 10889.2 |

The CSV also contains zero-angle-normalized survival ratios. These test an empirical
angle-independent noise-survival approximation; BB56 fit constants are not reused.

Costs include one Z-only initialization round, the rotation gadgets, and one full
terminal X/Z round. There is no physical early-abort saving in this protocol.
CNOT depth excludes reset, measurement and the single rotation layer. Qubit-layer
cost is an allocation proxy, not hardware-calibrated elapsed time.

Total sampled attempts in this report: 16,000,000.
Total sampling time: 23.126 seconds.
