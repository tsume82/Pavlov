## Comparison Table
| Function    | p(PPO mf wIntraDeltaF + IntraDeltaX < iDE) with AUC metric | p(PPO mf wIntraDeltaF + IntraDeltaX < iDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.3888 | **0.6172** |
| Discus | 0.4024 | **0.572** |
| Ellipsoid | 0.378 | **0.7056** |
| Katsuura | 0.4172 | 0.384 |
| Rastrigin | 0.3708 | **0.5628** |
| Rosenbrock | 0.3216 | **0.6204** |
| Schaffers | 0.4024 | **0.5292** |
| Schwefel | 0.2604 | **0.6492** |
| Sphere | 0.3516 | **0.5432** |
| Weierstrass | 0.4512 | 0.4316 |
| AttractiveSector_5D | 0.3584 | 0.4052 |
| AttractiveSector_10D | **0.526** | **0.7568** |
| AttractiveSector_20D | **0.8936** | **0.954** |
| BuecheRastrigin_5D | 0.3976 | 0.2192 |
| BuecheRastrigin_10D | **0.826** | **0.8552** |
| BuecheRastrigin_20D | 0.3104 | 0.3884 |
| CompositeGR_5D | 0.338 | 0.4288 |
| CompositeGR_10D | 0.1776 | 0.392 |
| CompositeGR_20D | 0.3292 | **0.6644** |
| DifferentPowers_5D | **0.5276** | **0.7528** |
| DifferentPowers_10D | 0.4532 | 0.3668 |
| DifferentPowers_20D | 0.1984 | 0.4228 |
| LinearSlope_5D | **0.5792** | **0.7608** |
| LinearSlope_10D | **0.67** | **0.9712** |
| LinearSlope_20D | 0.384 | **0.6612** |
| SharpRidge_5D | **0.6464** | **0.7628** |
| SharpRidge_10D | **0.9276** | **0.9932** |
| SharpRidge_20D | 0.3828 | **0.7388** |
| StepEllipsoidal_5D | **0.5372** | **0.6712** |
| StepEllipsoidal_10D | **0.7392** | **0.9724** |
| StepEllipsoidal_20D | **0.9136** | **0.91** |
| RosenbrockRotated_5D | 0.3224 | 0.4876 |
| RosenbrockRotated_10D | **0.6928** | **0.8688** |
| RosenbrockRotated_20D | 0.0984 | 0.37 |
| SchaffersIllConditioned_5D | 0.284 | 0.338 |
| SchaffersIllConditioned_10D | 0.2428 | 0.4592 |
| SchaffersIllConditioned_20D | 0.2288 | 0.3984 |
| LunacekBiR_5D | 0.4516 | 0.4332 |
| LunacekBiR_10D | 0.3828 | **0.5892** |
| LunacekBiR_20D | 0.1924 | 0.438 |
| GG101me_5D | **0.5256** | **0.5028** |
| GG101me_10D | **0.6904** | **0.8084** |
| GG101me_20D | 0.2268 | 0.3232 |
| GG21hi_5D | 0.4892 | 0.4884 |
| GG21hi_10D | 0.422 | **0.5784** |
| GG21hi_20D | 0.2796 | 0.4592 |
| **Total p(PPO mf wIntraDeltaF + IntraDeltaX < iDE)** | 30.4% (14/46) | **58.7**% (27/46) |

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