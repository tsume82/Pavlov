## Comparison Table
| Function    | p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < iDE) with AUC metric | p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < iDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.2812 | **0.5136** |
| Discus | 0.3584 | **0.6104** |
| Ellipsoid | 0.2888 | **0.6168** |
| Katsuura | 0.476 | 0.404 |
| Rastrigin | 0.3736 | **0.5876** |
| Rosenbrock | 0.2124 | **0.5796** |
| Schaffers | 0.4076 | **0.5724** |
| Schwefel | 0.3044 | **0.574** |
| Sphere | 0.2844 | **0.5164** |
| Weierstrass | **0.5776** | **0.5412** |
| AttractiveSector_5D | 0.3512 | 0.2036 |
| AttractiveSector_10D | **0.5204** | **0.7588** |
| AttractiveSector_20D | **0.876** | **0.9008** |
| BuecheRastrigin_5D | 0.2948 | 0.2064 |
| BuecheRastrigin_10D | **0.7836** | **0.874** |
| BuecheRastrigin_20D | 0.2624 | 0.4312 |
| CompositeGR_5D | 0.3112 | 0.3376 |
| CompositeGR_10D | 0.1476 | 0.3324 |
| CompositeGR_20D | 0.3872 | **0.724** |
| DifferentPowers_5D | **0.5252** | **0.6304** |
| DifferentPowers_10D | **0.5104** | 0.3964 |
| DifferentPowers_20D | 0.1312 | 0.3912 |
| LinearSlope_5D | **0.6008** | **0.7416** |
| LinearSlope_10D | **0.6784** | **0.9856** |
| LinearSlope_20D | 0.442 | **0.6888** |
| SharpRidge_5D | **0.6112** | **0.6808** |
| SharpRidge_10D | **0.9216** | **0.9992** |
| SharpRidge_20D | 0.246 | **0.6572** |
| StepEllipsoidal_5D | 0.4068 | **0.5876** |
| StepEllipsoidal_10D | **0.7164** | **0.9484** |
| StepEllipsoidal_20D | **0.898** | **0.974** |
| RosenbrockRotated_5D | 0.2952 | 0.4724 |
| RosenbrockRotated_10D | **0.646** | **0.7936** |
| RosenbrockRotated_20D | 0.0264 | 0.3112 |
| SchaffersIllConditioned_5D | 0.2564 | 0.272 |
| SchaffersIllConditioned_10D | 0.1292 | 0.3268 |
| SchaffersIllConditioned_20D | 0.1496 | 0.258 |
| LunacekBiR_5D | 0.3756 | 0.3904 |
| LunacekBiR_10D | 0.2148 | 0.3856 |
| LunacekBiR_20D | 0.108 | 0.402 |
| GG101me_5D | **0.5648** | 0.49 |
| GG101me_10D | **0.6848** | **0.7876** |
| GG101me_20D | 0.1488 | 0.2184 |
| GG21hi_5D | 0.4548 | 0.4352 |
| GG21hi_10D | 0.3132 | 0.4188 |
| GG21hi_20D | 0.2568 | 0.4504 |
| **Total p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < iDE)** | 32.6% (15/46) | **54.3**% (25/46) |

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