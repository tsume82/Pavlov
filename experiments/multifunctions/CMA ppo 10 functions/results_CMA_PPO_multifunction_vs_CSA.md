## Comparison Table
| Function    | p(PPO multifunction < CSA) with AUC metric | p(PPO multifunction < CSA) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.2608 | 0.3988 |
| Discus | 0.4076 | 0.0904 |
| Ellipsoid | 0.142 | 0.338 |
| Katsuura | 0.4396 | 0.4056 |
| Rastrigin | **1.0** | **0.9884** |
| Rosenbrock | 0.1668 | 0.3172 |
| Schaffers | **0.9208** | **0.8376** |
| Schwefel | **0.9716** | **0.5644** |
| Sphere | 0.13 | 0.1464 |
| Weierstrass | 0.3676 | 0.4476 |
| **Total p(PPO multifunction < CSA)** | 30.0% (3/10) | 30.0% (3/10) |

## Plots

##### BentCigar

![](imgs\PPO_multifunction_vs_CSA__BentCigar_comparison.png)

##### Discus

![](imgs\PPO_multifunction_vs_CSA__Discus_comparison.png)

##### Ellipsoid

![](imgs\PPO_multifunction_vs_CSA__Ellipsoid_comparison.png)

##### Katsuura

![](imgs\PPO_multifunction_vs_CSA__Katsuura_comparison.png)

##### Rastrigin

![](imgs\PPO_multifunction_vs_CSA__Rastrigin_comparison.png)

##### Rosenbrock

![](imgs\PPO_multifunction_vs_CSA__Rosenbrock_comparison.png)

##### Schaffers

![](imgs\PPO_multifunction_vs_CSA__Schaffers_comparison.png)

##### Schwefel

![](imgs\PPO_multifunction_vs_CSA__Schwefel_comparison.png)

##### Sphere

![](imgs\PPO_multifunction_vs_CSA__Sphere_comparison.png)

##### Weierstrass

![](imgs\PPO_multifunction_vs_CSA__Weierstrass_comparison.png)


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
    "env.env_class": "SchedulerPolicyMultiRayEnvironment",
    "env.env_config": {
        "solver_driver": "CMAdriver",
        "solver_driver_args": [
            [
                10,
                10,
                12,
                0.5
            ],
            [
                10,
                10,
                11,
                0.5
            ],
            [
                10,
                10,
                2,
                0.5
            ],
            [
                10,
                10,
                23,
                0.5
            ],
            [
                10,
                10,
                15,
                0.5
            ],
            [
                10,
                10,
                8,
                0.5
            ],
            [
                10,
                10,
                17,
                0.5
            ],
            [
                10,
                10,
                20,
                0.5
            ],
            [
                10,
                10,
                1,
                0.5
            ],
            [
                10,
                10,
                16,
                0.5
            ]
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