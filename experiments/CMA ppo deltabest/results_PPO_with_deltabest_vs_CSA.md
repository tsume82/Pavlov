## Comparison Table
| Function    | p(PPO with deltabest < CSA) with AUC metric | p(PPO with deltabest < CSA) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.2808 | **0.7736** |
| Discus | 0.2784 | 0.126 |
| Ellipsoid | 0.1256 | **0.6084** |
| Katsuura | 0.414 | 0.4532 |
| Rastrigin | **1.0** | **0.9984** |
| Rosenbrock | 0.2168 | **0.6032** |
| Schaffers | **0.8092** | **0.7616** |
| Schwefel | **0.9692** | 0.398 |
| Sphere | 0.094 | 0.4156 |
| Weierstrass | 0.4644 | **0.5176** |

## Plots

##### BentCigar

![](BentCigar/PPO_with_deltabest_vs_CSA:_BentCigar_comparison.png)

##### Discus

![](Discus/PPO_with_deltabest_vs_CSA:_Discus_comparison.png)

##### Ellipsoid

![](Ellipsoid/PPO_with_deltabest_vs_CSA:_Ellipsoid_comparison.png)

##### Katsuura

![](Katsuura/PPO_with_deltabest_vs_CSA:_Katsuura_comparison.png)

##### Rastrigin

![](Rastrigin/PPO_with_deltabest_vs_CSA:_Rastrigin_comparison.png)

##### Rosenbrock

![](Rosenbrock/PPO_with_deltabest_vs_CSA:_Rosenbrock_comparison.png)

##### Schaffers

![](Schaffers/PPO_with_deltabest_vs_CSA:_Schaffers_comparison.png)

##### Schwefel

![](Schwefel/PPO_with_deltabest_vs_CSA:_Schwefel_comparison.png)

##### Sphere

![](Sphere/PPO_with_deltabest_vs_CSA:_Sphere_comparison.png)

##### Weierstrass

![](Weierstrass/PPO_with_deltabest_vs_CSA:_Weierstrass_comparison.png)


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