## Comparison Table
| Function    | p(PPO with deltabest < PPO without deltabest) with AUC metric | p(PPO with deltabest < PPO without deltabest) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.2032 | **0.9348** |
| Discus | --- | --- |
| Ellipsoid | 0.2368 | **0.8932** |
| Katsuura | 0.4312 | 0.4076 |
| Rastrigin | **0.794** | **0.99** |
| Rosenbrock | --- | --- |
| Schaffers | --- | --- |
| Schwefel | **0.7404** | **0.6168** |
| Sphere | **0.9864** | **1.0** |
| Weierstrass | **0.9452** | **0.9912** |

## Plots

##### BentCigar

![](BentCigar/PPO_with_deltabest_vs_PPO_without_deltabest:_BentCigar_comparison.png)

##### Discus

![](Discus/PPO_with_deltabest_vs_PPO_without_deltabest:_Discus_comparison.png)

##### Ellipsoid

![](Ellipsoid/PPO_with_deltabest_vs_PPO_without_deltabest:_Ellipsoid_comparison.png)

##### Katsuura

![](Katsuura/PPO_with_deltabest_vs_PPO_without_deltabest:_Katsuura_comparison.png)

##### Rastrigin

![](Rastrigin/PPO_with_deltabest_vs_PPO_without_deltabest:_Rastrigin_comparison.png)

##### Rosenbrock

![](Rosenbrock/PPO_with_deltabest_vs_PPO_without_deltabest:_Rosenbrock_comparison.png)

##### Schaffers

![](Schaffers/PPO_with_deltabest_vs_PPO_without_deltabest:_Schaffers_comparison.png)

##### Schwefel

![](Schwefel/PPO_with_deltabest_vs_PPO_without_deltabest:_Schwefel_comparison.png)

##### Sphere

![](Sphere/PPO_with_deltabest_vs_PPO_without_deltabest:_Sphere_comparison.png)

##### Weierstrass

![](Weierstrass/PPO_with_deltabest_vs_PPO_without_deltabest:_Weierstrass_comparison.png)


## Configuration

```json
{
    "agent.algorithm": "RayProximalPolicyOptimization",
    "agent.algorithm.render_env": false,
    "agent.algorithm.num_workers": 0,
    "agent.algorithm.batch_mode": "complete_episodes",
    "agent.algorithm.lr": 1e-05,
    "agent.algorithm.train_batch_size": 200,
    "agent.algorithm.vf_clip_param": 10,
    "agent.algorithm.model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [
            50,
            50
        ]
    },
    "env.env_class": "SchedulerPolicyRayEnvironment",
    "env.env_config": {
        "solver_driver": "CMAdriver",
        "solver_driver_args": [
            10,
            10,
            12,
            0.5
        ],
        "maximize": false,
        "steps": 50,
        "state_metrics_names": [
            "DifferenceOfBest",
            "SolverStateHistory"
        ],
        "state_metrics_config": [
            [
                40,
                false,
                1,
                true,
                true
            ],
            [
                {
                    "step_size": {
                        "max": 3,
                        "min": 0
                    }
                },
                40
            ]
        ],
        "reward_metric": "DeltaBest",
        "reward_metric_config": [
            false,
            true,
            true
        ],
        "memes_no": 1,
        "action_space_config": {
            "step_size": {
                "max": 3,
                "min": 1e-05
            }
        }
    }
}
```