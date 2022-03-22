## Comparison Table
| Function    | p(PPO mf wIntraDeltaF + IntraDeltaX < iDE) with AUC metric | p(PPO mf wIntraDeltaF + IntraDeltaX < iDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.3284 | 0.4396 |
| Discus | 0.4688 | **0.7132** |
| Ellipsoid | 0.4072 | **0.518** |
| Katsuura | 0.432 | 0.4796 |
| Rastrigin | 0.3928 | **0.5796** |
| Rosenbrock | 0.254 | 0.4296 |
| Schaffers | 0.3752 | 0.4392 |
| Schwefel | 0.3588 | **0.562** |
| Sphere | 0.2464 | 0.4088 |
| Weierstrass | **0.7664** | **0.7744** |
| AttractiveSector_5D | 0.3748 | 0.2536 |
| AttractiveSector_10D | **0.6228** | **0.6216** |
| AttractiveSector_20D | **0.8944** | **0.7808** |
| BuecheRastrigin_5D | 0.31 | 0.1448 |
| BuecheRastrigin_10D | **0.6072** | **0.6228** |
| BuecheRastrigin_20D | 0.1644 | 0.144 |
| CompositeGR_5D | 0.3448 | 0.354 |
| CompositeGR_10D | 0.1796 | 0.4512 |
| CompositeGR_20D | 0.4724 | **0.7156** |
| DifferentPowers_5D | **0.5192** | **0.6696** |
| DifferentPowers_10D | 0.4184 | 0.1564 |
| DifferentPowers_20D | 0.1248 | 0.1828 |
| LinearSlope_5D | 0.4952 | **0.6844** |
| LinearSlope_10D | **0.5812** | **0.9408** |
| LinearSlope_20D | 0.2276 | 0.3696 |
| SharpRidge_5D | **0.554** | **0.614** |
| SharpRidge_10D | **0.8928** | **0.9828** |
| SharpRidge_20D | 0.1876 | 0.3572 |
| StepEllipsoidal_5D | **0.5664** | 0.4568 |
| StepEllipsoidal_10D | **0.7844** | **0.9168** |
| StepEllipsoidal_20D | **0.8092** | **0.7056** |
| RosenbrockRotated_5D | 0.3212 | 0.4196 |
| RosenbrockRotated_10D | **0.694** | **0.7144** |
| RosenbrockRotated_20D | 0.1756 | 0.1928 |
| SchaffersIllConditioned_5D | 0.2912 | 0.2664 |
| SchaffersIllConditioned_10D | 0.1996 | 0.25 |
| SchaffersIllConditioned_20D | 0.208 | 0.2916 |
| LunacekBiR_5D | 0.4316 | **0.516** |
| LunacekBiR_10D | 0.3448 | **0.5532** |
| LunacekBiR_20D | 0.214 | 0.3488 |
| GG101me_5D | **0.5652** | **0.5312** |
| GG101me_10D | **0.7208** | **0.73** |
| GG101me_20D | 0.1636 | 0.1492 |
| GG21hi_5D | **0.5632** | 0.4292 |
| GG21hi_10D | 0.2792 | 0.3396 |
| GG21hi_20D | 0.2764 | 0.3728 |
| **Total p(PPO mf wIntraDeltaF + IntraDeltaX < iDE)** | 32.6% (15/46) | 45.7% (21/46) |

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