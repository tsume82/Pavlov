## Comparison Table
| Function    | p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < jDE) with AUC metric | p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < jDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.4044 | **0.5716** |
| Discus | 0.432 | 0.2932 |
| Ellipsoid | 0.39 | **0.5964** |
| Katsuura | **0.6172** | **0.5752** |
| Rastrigin | 0.3364 | 0.4712 |
| Rosenbrock | 0.4228 | **0.5132** |
| Schaffers | 0.354 | 0.4856 |
| Schwefel | 0.3076 | 0.3656 |
| Sphere | 0.2112 | **0.5068** |
| Weierstrass | 0.4844 | 0.4856 |
| AttractiveSector_5D | 0.4828 | **0.59** |
| AttractiveSector_10D | 0.3944 | **0.5516** |
| AttractiveSector_20D | 0.1924 | **0.5164** |
| BuecheRastrigin_5D | 0.2952 | **0.5396** |
| BuecheRastrigin_10D | 0.3956 | **0.6068** |
| BuecheRastrigin_20D | 0.3532 | **0.6068** |
| CompositeGR_5D | 0.2924 | 0.4332 |
| CompositeGR_10D | 0.236 | 0.382 |
| CompositeGR_20D | 0.1428 | 0.3056 |
| DifferentPowers_5D | 0.2484 | **0.6424** |
| DifferentPowers_10D | 0.362 | **0.6168** |
| DifferentPowers_20D | 0.344 | **0.684** |
| LinearSlope_5D | 0.36 | 0.2004 |
| LinearSlope_10D | 0.456 | **0.6856** |
| LinearSlope_20D | 0.358 | **0.6356** |
| SharpRidge_5D | 0.262 | **0.5444** |
| SharpRidge_10D | 0.398 | **0.6892** |
| SharpRidge_20D | 0.3684 | **0.6696** |
| StepEllipsoidal_5D | 0.3764 | **0.6848** |
| StepEllipsoidal_10D | 0.4928 | **0.8508** |
| StepEllipsoidal_20D | 0.3604 | **0.6508** |
| RosenbrockRotated_5D | 0.3356 | **0.5704** |
| RosenbrockRotated_10D | 0.2188 | **0.5164** |
| RosenbrockRotated_20D | 0.1736 | 0.4624 |
| SchaffersIllConditioned_5D | 0.3668 | **0.6412** |
| SchaffersIllConditioned_10D | 0.3732 | **0.5436** |
| SchaffersIllConditioned_20D | 0.3284 | 0.45 |
| LunacekBiR_5D | 0.3056 | 0.3568 |
| LunacekBiR_10D | 0.2664 | 0.4176 |
| LunacekBiR_20D | 0.1932 | 0.4408 |
| GG101me_5D | 0.452 | **0.6348** |
| GG101me_10D | 0.3996 | **0.6032** |
| GG101me_20D | 0.2992 | 0.4744 |
| GG21hi_5D | 0.4124 | 0.4916 |
| GG21hi_10D | 0.4256 | **0.6072** |
| GG21hi_20D | 0.4028 | **0.5768** |
| **Total p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < jDE)** | 2.2% (1/46) | **65.2**% (30/46) |

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
            50,
            50
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