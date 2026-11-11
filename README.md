# Pesquisa em otimização de rotas: ML clássico, quântico e híbrido

Este repositório reúne os projetos de P&D em roteamento de veículos que desenvolvi no Instituto Eldorado. Os projetos foram descontinuados pela empresa e o código é publicado com autorização. Cada pasta tem um README próprio com o problema, a abordagem e como reproduzir.

| Pasta | Projeto | Destaques |
|---|---|---|
| [`qgat/`](qgat/) | **QGAT**: Graph Attention com camadas quânticas híbridas, treinado por RL (PPO e REINFORCE) para CVRP | Um único entry point para local, Slurm e AWS SageMaker; MLflow; 49 configs de ablação; pytest |
| [`qaoa-multiobjective/`](qaoa-multiobjective/) | **QAOA multiobjetivo** para TSP e VRP | Formulações QUBO multiobjetivo, execuções em hardware IBM, hipervolume da fronteira de Pareto |

O dataset desses modelos foi coletado com o [delivery-graph-collector](../delivery-graph-collector), que mede tempos de trajeto reais via Google Routes API com orçamento controlado. <!-- TODO: trocar pelo link do GitHub -->

## Publicações

| Trabalho | Autores | Venue / status | Pasta |
|---|---|---|---|
| *Quantum Approximate Multi-Objective Optimization in Routing Problems* | Lussi, Fumaco, **Reballo**, Libois Neto, de Barros, Duzzioni | IOP (formato do manuscrito) <!-- TODO: status + DOI/arXiv --> | `qaoa-multiobjective/` |
| <!-- TODO: demais papers --> | | | |
