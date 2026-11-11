# QAOA multiobjetivo para problemas de roteamento

Este é o código do paper **"Quantum Approximate Multi-Objective Optimization in Routing Problems"**, publicado em [arXiv:2610.00168](https://arxiv.org/abs/2610.00168) em 2026. Os autores são E. W. Lussi, A. P. Fumaco, **M. V. Reballo**, J. C. Libois Neto, F. A. C. de Barros e E. I. Duzzioni, do Instituto Eldorado, da UFSC e da Quantuloop.

## O problema

Uma operação de roteamento real quase nunca tem um objetivo único. Ela precisa equilibrar distância, tempo, custo e carga ao mesmo tempo. O paper estende o *Quantum Approximate Multi-Objective Optimization* (QAMOO) para TSP e VRP. Para isso, formula esses problemas como QUBOs multiobjetivo e mede a qualidade da fronteira de Pareto que o QAOA encontra. A métrica é o hipervolume, calculado em simulação MPS e em hardware IBM real.

## O que está aqui

| Pasta / arquivo | Conteúdo |
|---|---|
| `vrp/` | Formulações QUBO: `TSP`, `VRP_QUBO`, `CVRP_TimeIndexed_QUBO`, `MOVRP_QUBO`, `MOVRP_TimeIndexed_QUBO`, `MOCVRP_TimeIndexed_QUBO` e QAOA com cuQuantum |
| `extract_hw_data.py`, `extract_hw_plots_v2.py`, `compute_hv_hardware.py` | Extração dos resultados das execuções em hardware IBM (TSP 17→37 e VRP 27→43 qubits) e cálculo do hipervolume |
| `notebooks/`, `parameter_training/` | Experimentos de base derivados do QAMOO original (QAOA, Goemans-Williamson, soma ponderada, ε-restrição e treino de ângulos com JuliQAOA) |
| `figures/` | Figuras de resultados (simulação MPS, hardware, hipervolume e pontos não dominados) |

## Dependências e créditos

Os notebooks de base e os utilitários de resultado vêm do repositório [qamoo](https://github.com/stefan-woerner/qamoo), da IBM Research. Para rodar, clone-o na raiz desta pasta (`git clone https://github.com/stefan-woerner/qamoo.git`). O treino de ângulos usa o [JuliQAOA](https://github.com/lanl/JuliQAOA.jl), do LANL.

As instâncias de problema, que somam cerca de 7 GB, não estão no repositório. <!-- TODO: link para dataset, se houver -->
