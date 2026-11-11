# Pesquisa em otimização de rotas: ML clássico, quântico e híbrido

Este repositório reúne os projetos de P&D em roteamento de veículos que desenvolvi no Instituto Eldorado. Os projetos foram descontinuados pela empresa e o código é publicado com autorização. Cada pasta tem um README próprio com o problema, a abordagem e como reproduzir.

| Pasta | Projeto | Destaques |
|---|---|---|
| [`qgat/`](qgat/) | **QGAT**: Graph Attention com camadas quânticas híbridas, treinado por RL (PPO e REINFORCE) para CVRP | Um único entry point para local, Slurm e AWS SageMaker; MLflow; 49 configs de ablação; pytest |
| [`qaoa-multiobjective/`](qaoa-multiobjective/) | **QAOA multiobjetivo** para TSP e VRP | Formulações QUBO multiobjetivo, execuções em hardware IBM, hipervolume da fronteira de Pareto |

O dataset desses modelos foi coletado com o [delivery-graph-collector](https://github.com/marcosreballo-source/delivery-graph-collector), que mede tempos de trajeto reais via Google Routes API com orçamento controlado.

## Publicações

| Trabalho | Autores | Onde | Código |
|---|---|---|---|
| *A hybrid quantum-classical neural network for learning to route* (2026) | Ritt, da Rosa Júnior, **Reballo**, do Amaral, de Barros | [arXiv:2609.00489](https://arxiv.org/abs/2609.00489), submetido ao CBCTQ 2026 | [`qgat/`](qgat/) |
| *Quantum Approximate Multi-Objective Optimization in Routing Problems* (2026) | Lussi, Fumaco, **Reballo**, Libois Neto, de Barros, Duzzioni | [arXiv:2610.00168](https://arxiv.org/abs/2610.00168) | [`qaoa-multiobjective/`](qaoa-multiobjective/) |
| *Evaluating Parameter Transfer in FALQON Across Graph Families* (2026) | Fumaco, **Reballo**, de Barros, Thomaz, Duzzioni | 1º Simpósio Brasileiro de Computação e Comunicação Quânticas (SBCCQ), [DOI 10.5753/sbccq.2026.23365](https://doi.org/10.5753/sbccq.2026.23365), [arXiv:2605.29917](https://arxiv.org/abs/2605.29917) | não incluído |
| *Sparse Walsh-Pauli Correlation Encoding for MaxCut: A Quantum-Inspired Formulation with Quantum-Hardware Validation* (2026) | do Amaral, Legnazzi, **Reballo**, Ritt, da Rosa Júnior, de Barros | [arXiv:2609.08907](https://arxiv.org/abs/2609.08907) | não incluído |

Perfil ORCID: [0009-0003-6396-9302](https://orcid.org/0009-0003-6396-9302)
