## Comparison Table
| Function    | p(PPO mf wIntraDeltaF + IntraDeltaX < jDE) with AUC metric | p(PPO mf wIntraDeltaF + IntraDeltaX < jDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.2732 | **0.53** |
| Discus | 0.4972 | 0.346 |
| Ellipsoid | 0.2696 | 0.4948 |
| Katsuura | **0.57** | **0.5848** |
| Rastrigin | 0.41 | **0.6108** |
| Rosenbrock | 0.3508 | **0.5272** |
| Schaffers | 0.358 | 0.466 |
| Schwefel | 0.3596 | 0.4592 |
| Sphere | 0.1988 | 0.4384 |
| Weierstrass | **0.5588** | **0.6304** |
| AttractiveSector_5D | 0.4376 | **0.5992** |
| AttractiveSector_10D | 0.3292 | 0.4448 |
| AttractiveSector_20D | 0.1696 | 0.4088 |
| BuecheRastrigin_5D | 0.376 | **0.5308** |
| BuecheRastrigin_10D | 0.3652 | **0.582** |
| BuecheRastrigin_20D | 0.2408 | 0.4984 |
| CompositeGR_5D | 0.2712 | 0.4988 |
| CompositeGR_10D | 0.236 | 0.3944 |
| CompositeGR_20D | 0.23 | 0.4112 |
| DifferentPowers_5D | 0.2552 | **0.61** |
| DifferentPowers_10D | 0.4124 | **0.6152** |
| DifferentPowers_20D | 0.2876 | **0.5656** |
| LinearSlope_5D | 0.3148 | 0.2092 |
| LinearSlope_10D | 0.4972 | **0.622** |
| LinearSlope_20D | 0.3144 | 0.4616 |
| SharpRidge_5D | 0.2544 | **0.516** |
| SharpRidge_10D | 0.3852 | **0.64** |
| SharpRidge_20D | 0.3232 | 0.4956 |
| StepEllipsoidal_5D | 0.4004 | **0.6648** |
| StepEllipsoidal_10D | 0.4356 | **0.7632** |
| StepEllipsoidal_20D | 0.292 | **0.564** |
| RosenbrockRotated_5D | 0.3428 | **0.5792** |
| RosenbrockRotated_10D | 0.2172 | **0.542** |
| RosenbrockRotated_20D | 0.1872 | 0.3264 |
| SchaffersIllConditioned_5D | 0.33 | **0.552** |
| SchaffersIllConditioned_10D | 0.3868 | **0.5524** |
| SchaffersIllConditioned_20D | 0.4152 | **0.5464** |
| LunacekBiR_5D | 0.3408 | 0.478 |
| LunacekBiR_10D | 0.3616 | **0.5116** |
| LunacekBiR_20D | 0.2608 | 0.42 |
| GG101me_5D | **0.5252** | **0.634** |
| GG101me_10D | 0.3764 | **0.5192** |
| GG101me_20D | 0.2696 | 0.3512 |
| GG21hi_5D | 0.4392 | **0.5432** |
| GG21hi_10D | 0.4 | **0.5396** |
| GG21hi_20D | 0.4628 | **0.5012** |
| **Total p(PPO mf wIntraDeltaF + IntraDeltaX < jDE)** | 6.5% (3/46) | **60.9**% (28/46) |

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