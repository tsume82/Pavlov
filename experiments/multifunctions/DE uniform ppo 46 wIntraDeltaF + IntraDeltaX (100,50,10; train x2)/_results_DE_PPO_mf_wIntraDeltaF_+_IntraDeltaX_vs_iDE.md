## Comparison Table
| Function    | p(PPO mf wIntraDeltaF + IntraDeltaX < iDE) with AUC metric | p(PPO mf wIntraDeltaF + IntraDeltaX < iDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.2492 | 0.4024 |
| Discus | 0.4832 | **0.6704** |
| Ellipsoid | 0.2488 | 0.442 |
| Katsuura | 0.474 | 0.432 |
| Rastrigin | 0.272 | **0.6584** |
| Rosenbrock | 0.22 | 0.374 |
| Schaffers | 0.3052 | **0.5388** |
| Schwefel | 0.2368 | 0.4796 |
| Sphere | 0.2092 | 0.4692 |
| Weierstrass | 0.4844 | **0.56** |
| AttractiveSector_5D | 0.4756 | 0.206 |
| AttractiveSector_10D | **0.5368** | **0.5512** |
| AttractiveSector_20D | **0.826** | **0.7916** |
| BuecheRastrigin_5D | 0.3292 | 0.1996 |
| BuecheRastrigin_10D | **0.6776** | **0.74** |
| BuecheRastrigin_20D | 0.1704 | 0.4088 |
| CompositeGR_5D | 0.358 | 0.3492 |
| CompositeGR_10D | 0.1496 | 0.3872 |
| CompositeGR_20D | 0.292 | **0.6124** |
| DifferentPowers_5D | **0.57** | **0.7188** |
| DifferentPowers_10D | 0.4264 | 0.3308 |
| DifferentPowers_20D | 0.1356 | 0.3088 |
| LinearSlope_5D | **0.538** | **0.7236** |
| LinearSlope_10D | **0.6208** | **0.928** |
| LinearSlope_20D | 0.3448 | **0.6188** |
| SharpRidge_5D | 0.4076 | **0.5748** |
| SharpRidge_10D | **0.9032** | **0.996** |
| SharpRidge_20D | 0.3128 | **0.7052** |
| StepEllipsoidal_5D | **0.5** | **0.5708** |
| StepEllipsoidal_10D | **0.6424** | **0.8732** |
| StepEllipsoidal_20D | **0.8272** | **0.788** |
| RosenbrockRotated_5D | 0.264 | 0.396 |
| RosenbrockRotated_10D | **0.5276** | **0.672** |
| RosenbrockRotated_20D | 0.0308 | 0.2096 |
| SchaffersIllConditioned_5D | 0.1916 | 0.2636 |
| SchaffersIllConditioned_10D | 0.152 | 0.322 |
| SchaffersIllConditioned_20D | 0.1444 | 0.2584 |
| LunacekBiR_5D | 0.3624 | 0.4032 |
| LunacekBiR_10D | 0.2384 | **0.548** |
| LunacekBiR_20D | 0.1504 | 0.4208 |
| GG101me_5D | 0.4688 | 0.4448 |
| GG101me_10D | **0.668** | **0.754** |
| GG101me_20D | 0.1964 | 0.2544 |
| GG21hi_5D | 0.4772 | 0.4456 |
| GG21hi_10D | 0.3768 | 0.482 |
| GG21hi_20D | 0.3028 | 0.4988 |
| **Total p(PPO mf wIntraDeltaF + IntraDeltaX < iDE)** | 23.9% (11/46) | 45.7% (21/46) |

## Configuration

```json
{
    "agent.algorithm": "RayProximalPolicyOptimization",
    "agent.algorithm.render_env": false,
    "agent.algorithm.num_workers": 0,
    "agent.algorithm.batch_mode": "complete_episodes",
    "agent.algorithm.lr": 5e-05,
    "agent.algorithm.train_batch_size": 200,
    "agent.algorithm.optimizer": "Adam",
    "agent.algorithm.vf_clip_param": 10,
    "agent.algorithm.model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [
            100,
            50,
            10
        ]
    },
    "env.env_class": "SchedulerPolicyMultiRayEnvironment",
    "env.env_config": {
        "solver_driver": "DEdriver",
        "solver_driver_args": [
            [
                10,
                10,
                12,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                11,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                2,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                23,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                15,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                8,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                17,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                20,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                1,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                16,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                6,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                6,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                6,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                4,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                4,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                4,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                19,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                19,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                19,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                14,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                14,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                14,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                5,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                5,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                5,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                13,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                13,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                13,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                7,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                7,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                7,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                9,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                9,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                9,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                18,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                18,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                18,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                24,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                24,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                24,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                21,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                21,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                21,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                22,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                22,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                22,
                "best1bin",
                "uniform"
            ]
        ],
        "maximize": false,
        "steps": 50,
        "state_metrics_names": [
            "DifferenceOfBest",
            "DeltaFitPop",
            "DeltaX",
            "SolverStateHistory"
        ],
        "state_metrics_config": [
            [
                40,
                false,
                1,
                true,
                false
            ],
            [
                40,
                false
            ],
            [
                40,
                false,
                true
            ],
            [
                {
                    "F_min": {
                        "max": [
                            2
                        ],
                        "min": [
                            0
                        ]
                    },
                    "F_max": {
                        "max": [
                            2
                        ],
                        "min": [
                            0
                        ]
                    },
                    "CR_min": {
                        "max": [
                            1
                        ],
                        "min": [
                            0
                        ]
                    },
                    "CR_max": {
                        "max": [
                            1
                        ],
                        "min": [
                            0
                        ]
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
            "F_min": {
                "max": 2,
                "min": 0
            },
            "F_max": {
                "max": 2,
                "min": 0
            },
            "CR_min": {
                "max": 1,
                "min": 0
            },
            "CR_max": {
                "max": 1,
                "min": 0
            }
        }
    }
}
```