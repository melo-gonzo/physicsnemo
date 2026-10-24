# Tensile2d: training-path effect on four seeds, and R1 under one function (eager, native)

| field | bug-trained 42 / 43 / 44 / 45 (mean; σ log) | fix-trained 42 / 43 / 44 / 45 (mean; σ log) | fix ÷ bug (seed means) | seed ranges | SE of the log ratio |
|---|---|---|---|---|---|
| displacement | 0.00729 / 0.00697 / 0.00737 / 0.00708 (0.00717; 0.025) | 0.00734 / 0.00644 / 0.00908 / 0.00867 (0.00788; 0.158) | 1.099 | overlap | 0.080 |
| stress | 0.00900 / 0.00871 / 0.00953 / 0.00938 (0.00915; 0.040) | 0.00938 / 0.00821 / 0.01024 / 0.01155 (0.00985; 0.144) | 1.075 | overlap | 0.075 |
| plastic strain | 0.34283 / 0.23949 / 0.25657 / 0.19299 (0.25797; 0.237) | 0.20261 / 0.16365 / 0.20106 / 0.23540 (0.20068; 0.149) | 0.778 | overlap | 0.140 |

R1 on Tensile2d (0.9.0 ÷ A2, seed means over seeds 42 / 43 unless stated; q = new seed ÷ A2 seed mean):

| comparison | displacement r (q42, q43) | stress r | plastic strain r |
|---|---|---|---|
| registered scorer numbers (both bug-trained, compiled path) | 1.041 (1.053, 1.030) | 1.135 | 0.691 |
| both bug-trained, eager | 1.045 (1.068, 1.022) | 1.135 | 0.716 |
| both fix-trained, eager (one function) | 0.831 (0.885, 0.777) | 0.944 | 0.858 |
| fix-trained 0.9.0 four seeds ÷ fix-trained A2, eager | 0.951 (0.885, 0.777, 1.095, 1.046) | 1.057 | 0.940 |

A2 seeds (eager): bug-trained displacement 0.00679 / 0.00685, stress 0.00709 / 0.00851, plastic strain 0.36762 / 0.44551
A2 seeds (eager): fix-trained displacement 0.00836 / 0.00823, stress 0.00919 / 0.00944, plastic strain 0.18683 / 0.24016
A2 training-path effect T_A2 = a2rfix ÷ a2r (eager, seed means): displacement 1.216, stress 1.194, plastic strain 0.525
posed ÷ native: a2rfix s42 1.0672, s43 1.0949; r090fix s42 1.0005, s43 1.0000, s44 1.0003, s45 1.0001 (displacement)
