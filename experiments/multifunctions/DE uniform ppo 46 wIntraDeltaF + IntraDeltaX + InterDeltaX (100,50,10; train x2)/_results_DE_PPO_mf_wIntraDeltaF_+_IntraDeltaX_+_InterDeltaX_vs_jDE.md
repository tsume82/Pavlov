## Comparison Table
| Function    | p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < jDE) with AUC metric | p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < jDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.2544 | 0.49 |
| Discus | 0.3932 | 0.3552 |
| Ellipsoid | 0.246 | 0.4128 |
| Katsuura | **0.6032** | **0.5** |
| Rastrigin | 0.3592 | 0.4928 |
| Rosenbrock | 0.2788 | 0.4952 |
| Schaffers | 0.3348 | 0.46 |
| Schwefel | 0.2904 | 0.2376 |
| Sphere | 0.1452 | 0.388 |
| Weierstrass | **0.5332** | 0.4964 |
| AttractiveSector_5D | 0.432 | 0.3552 |
| AttractiveSector_10D | 0.3272 | 0.4764 |
| AttractiveSector_20D | 0.1536 | 0.4648 |
| BuecheRastrigin_5D | 0.2556 | **0.5** |
| BuecheRastrigin_10D | 0.3364 | **0.5192** |
| BuecheRastrigin_20D | 0.2608 | **0.5544** |
| CompositeGR_5D | 0.2812 | 0.3516 |
| CompositeGR_10D | 0.1656 | 0.3112 |
| CompositeGR_20D | 0.1704 | 0.3888 |
| DifferentPowers_5D | 0.1784 | 0.4656 |
| DifferentPowers_10D | 0.3648 | **0.5608** |
| DifferentPowers_20D | 0.262 | **0.636** |
| LinearSlope_5D | 0.418 | 0.2012 |
| LinearSlope_10D | 0.476 | **0.7388** |
| LinearSlope_20D | 0.3864 | **0.5492** |
| SharpRidge_5D | 0.2508 | **0.5068** |
| SharpRidge_10D | 0.3272 | **0.6316** |
| SharpRidge_20D | 0.2304 | **0.5752** |
| StepEllipsoidal_5D | 0.304 | **0.6336** |
| StepEllipsoidal_10D | 0.3428 | **0.734** |
| StepEllipsoidal_20D | 0.2164 | **0.59** |
| RosenbrockRotated_5D | 0.2744 | **0.5436** |
| RosenbrockRotated_10D | 0.196 | 0.474 |
| RosenbrockRotated_20D | 0.058 | 0.3196 |
| SchaffersIllConditioned_5D | 0.3252 | 0.4464 |
| SchaffersIllConditioned_10D | 0.2932 | 0.4856 |
| SchaffersIllConditioned_20D | 0.25 | 0.3916 |
| LunacekBiR_5D | 0.2644 | 0.2872 |
| LunacekBiR_10D | 0.2136 | 0.33 |
| LunacekBiR_20D | 0.164 | 0.43 |
| GG101me_5D | 0.4952 | **0.5752** |
| GG101me_10D | 0.2688 | 0.4732 |
| GG101me_20D | 0.1924 | 0.3428 |
| GG21hi_5D | 0.3968 | 0.482 |
| GG21hi_10D | 0.3084 | 0.452 |
| GG21hi_20D | 0.3536 | 0.468 |
| **Total p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < jDE)** | 4.3% (2/46) | 30.4% (14/46) |

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
                40,
                false,
                false
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