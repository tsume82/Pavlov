## Comparison Table
| Function    | p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < iDE) with AUC metric | p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < iDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.2956 | 0.4644 |
| Discus | 0.432 | **0.72** |
| Ellipsoid | 0.26 | **0.536** |
| Katsuura | 0.39 | 0.3672 |
| Rastrigin | 0.234 | 0.464 |
| Rosenbrock | 0.1952 | 0.498 |
| Schaffers | 0.3924 | 0.4836 |
| Schwefel | 0.328 | **0.5348** |
| Sphere | 0.2464 | 0.4436 |
| Weierstrass | **0.7136** | **0.6892** |
| AttractiveSector_5D | 0.4032 | 0.2608 |
| AttractiveSector_10D | **0.5132** | **0.6628** |
| AttractiveSector_20D | **0.8136** | **0.7756** |
| BuecheRastrigin_5D | 0.346 | 0.1712 |
| BuecheRastrigin_10D | **0.7188** | **0.8368** |
| BuecheRastrigin_20D | 0.18 | 0.2008 |
| CompositeGR_5D | 0.3572 | 0.3504 |
| CompositeGR_10D | 0.1932 | 0.336 |
| CompositeGR_20D | 0.3392 | **0.59** |
| DifferentPowers_5D | **0.5688** | **0.7212** |
| DifferentPowers_10D | 0.3892 | 0.1956 |
| DifferentPowers_20D | 0.0748 | 0.1744 |
| LinearSlope_5D | **0.5556** | **0.7104** |
| LinearSlope_10D | **0.5684** | **0.9388** |
| LinearSlope_20D | 0.2184 | 0.456 |
| SharpRidge_5D | **0.582** | **0.6536** |
| SharpRidge_10D | **0.8624** | **0.9788** |
| SharpRidge_20D | 0.2236 | **0.5376** |
| StepEllipsoidal_5D | 0.4356 | **0.5512** |
| StepEllipsoidal_10D | **0.6888** | **0.8908** |
| StepEllipsoidal_20D | **0.84** | **0.812** |
| RosenbrockRotated_5D | 0.362 | 0.4316 |
| RosenbrockRotated_10D | **0.6772** | **0.8336** |
| RosenbrockRotated_20D | 0.0824 | 0.18 |
| SchaffersIllConditioned_5D | 0.2644 | 0.2796 |
| SchaffersIllConditioned_10D | 0.1088 | 0.1676 |
| SchaffersIllConditioned_20D | 0.1636 | 0.2148 |
| LunacekBiR_5D | 0.4436 | 0.4588 |
| LunacekBiR_10D | 0.2356 | 0.3748 |
| LunacekBiR_20D | 0.1144 | 0.2736 |
| GG101me_5D | **0.5536** | 0.4912 |
| GG101me_10D | **0.75** | **0.7804** |
| GG101me_20D | 0.1872 | 0.2308 |
| GG21hi_5D | 0.4484 | 0.422 |
| GG21hi_10D | 0.2884 | 0.3716 |
| GG21hi_20D | 0.2392 | 0.3216 |
| **Total p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < iDE)** | 30.4% (14/46) | 41.3% (19/46) |

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
                "normal"
            ],
            [
                10,
                10,
                11,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                2,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                23,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                15,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                8,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                17,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                20,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                1,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                16,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                6,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                6,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                6,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                4,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                4,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                4,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                19,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                19,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                19,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                14,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                14,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                14,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                5,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                5,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                5,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                13,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                13,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                13,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                7,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                7,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                7,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                9,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                9,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                9,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                18,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                18,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                18,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                24,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                24,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                24,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                21,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                21,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                21,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                22,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                22,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                22,
                "best1bin",
                "normal"
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
                    "F_mean": {
                        "max": [
                            2
                        ],
                        "min": [
                            0
                        ]
                    },
                    "F_stdev": {
                        "max": [
                            1
                        ],
                        "min": [
                            0
                        ]
                    },
                    "CR_mean": {
                        "max": [
                            1
                        ],
                        "min": [
                            0
                        ]
                    },
                    "CR_stdev": {
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
            "F_mean": {
                "max": 2,
                "min": 0
            },
            "F_stdev": {
                "max": 1,
                "min": 0
            },
            "CR_mean": {
                "max": 1,
                "min": 0
            },
            "CR_stdev": {
                "max": 1,
                "min": 0
            }
        }
    }
}
```