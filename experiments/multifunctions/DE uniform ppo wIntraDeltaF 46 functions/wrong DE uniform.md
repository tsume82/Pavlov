# DE unifrom PPO mf wDeltaFitPop vs jDE
| Function    | p(PPO mf wDeltaFitPop < jDE) with AUC metric | p(PPO mf wDeltaFitPop < jDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.346 | **0.6032** |
| Discus | 0.4376 | 0.356 |
| Ellipsoid | 0.298 | 0.4648 |
| Katsuura | **0.6332** | **0.5488** |
| Rastrigin | 0.3196 | **0.5532** |
| Rosenbrock | 0.3104 | 0.4592 |
| Schaffers | 0.2888 | 0.4384 |
| Schwefel | 0.2544 | 0.3452 |
| Sphere | 0.108 | 0.3124 |
| Weierstrass | 0.446 | 0.4236 |
| AttractiveSector_5D | **0.5968** | **0.996** |
| AttractiveSector_10D | 0.31 | **0.582** |
| AttractiveSector_20D | 0.0 | 0.0076 |
| BuecheRastrigin_5D | **1.0** | **0.9988** |
| BuecheRastrigin_10D | 0.26 | **0.5072** |
| BuecheRastrigin_20D | 0.0 | 0.0056 |
| CompositeGR_5D | **0.9988** | **0.9776** |
| CompositeGR_10D | 0.2736 | 0.4624 |
| CompositeGR_20D | 0.0 | 0.0104 |
| DifferentPowers_5D | **0.9936** | **0.9708** |
| DifferentPowers_10D | 0.3028 | **0.65** |
| DifferentPowers_20D | 0.0008 | 0.0496 |
| LinearSlope_5D | **1.0** | **0.9848** |
| LinearSlope_10D | **0.5744** | **0.6444** |
| LinearSlope_20D | 0.0 | 0.0 |
| SharpRidge_5D | **0.994** | **0.9828** |
| SharpRidge_10D | 0.44 | **0.6352** |
| SharpRidge_20D | 0.0 | 0.0124 |
| StepEllipsoidal_5D | **0.9956** | **0.9872** |
| StepEllipsoidal_10D | 0.308 | **0.6852** |
| StepEllipsoidal_20D | 0.0012 | 0.0892 |
| RosenbrockRotated_5D | **0.9888** | **0.9524** |
| RosenbrockRotated_10D | 0.2128 | **0.5864** |
| RosenbrockRotated_20D | 0.0 | 0.0128 |
| SchaffersIllConditioned_5D | **0.9408** | **0.9592** |
| SchaffersIllConditioned_10D | 0.2884 | **0.5552** |
| SchaffersIllConditioned_20D | 0.0144 | 0.0592 |
| LunacekBiR_5D | **1.0** | **1.0** |
| LunacekBiR_10D | 0.2984 | 0.4248 |
| LunacekBiR_20D | 0.0 | 0.0 |
| GG101me_5D | **0.9608** | **0.8088** |
| GG101me_10D | 0.368 | **0.5432** |
| GG101me_20D | 0.0332 | 0.1968 |
| GG21hi_5D | **0.8748** | **0.8772** |
| GG21hi_10D | 0.3688 | **0.5612** |
| GG21hi_20D | 0.0108 | 0.1184 |
| **Total p(PPO mf wDeltaFitPop < jDE)** | 30.4% (14/46) | **54.3**% (25/46) |

# DE unifrom PPO mf wDeltaFitPop vs iDE

| Function    | p(PPO mf wDeltaFitPop < iDE) with AUC metric | p(PPO mf wDeltaFitPop < iDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.386 | **0.6388** |
| Discus | 0.4032 | **0.6532** |
| Ellipsoid | 0.3376 | **0.6628** |
| Katsuura | **0.528** | 0.4588 |
| Rastrigin | 0.3304 | **0.6548** |
| Rosenbrock | 0.2556 | **0.5596** |
| Schaffers | 0.352 | **0.538** |
| Schwefel | 0.2744 | **0.6392** |
| Sphere | 0.2436 | 0.4684 |
| Weierstrass | 0.4832 | 0.4788 |
| AttractiveSector_5D | **0.5652** | **0.9992** |
| AttractiveSector_10D | 0.444 | **0.8188** |
| AttractiveSector_20D | 0.0 | 0.0008 |
| BuecheRastrigin_5D | **1.0** | **1.0** |
| BuecheRastrigin_10D | 0.2736 | 0.3796 |
| BuecheRastrigin_20D | 0.0156 | 0.3384 |
| CompositeGR_5D | **1.0** | **0.998** |
| CompositeGR_10D | 0.1572 | 0.3772 |
| CompositeGR_20D | 0.0416 | 0.05 |
| DifferentPowers_5D | **1.0** | **0.9888** |
| DifferentPowers_10D | 0.3536 | **0.8012** |
| DifferentPowers_20D | 0.0004 | 0.0036 |
| LinearSlope_5D | **1.0** | **0.9632** |
| LinearSlope_10D | 0.4232 | 0.2688 |
| LinearSlope_20D | 0.0 | 0.0076 |
| SharpRidge_5D | **1.0** | **0.9952** |
| SharpRidge_10D | 0.2272 | 0.418 |
| SharpRidge_20D | 0.0 | 0.0 |
| StepEllipsoidal_5D | **0.9992** | **0.9392** |
| StepEllipsoidal_10D | 0.3428 | **0.8072** |
| StepEllipsoidal_20D | 0.0 | 0.0068 |
| RosenbrockRotated_5D | **1.0** | **1.0** |
| RosenbrockRotated_10D | **0.7224** | **0.8312** |
| RosenbrockRotated_20D | 0.0 | 0.0072 |
| SchaffersIllConditioned_5D | **0.9516** | **1.0** |
| SchaffersIllConditioned_10D | 0.2148 | 0.4068 |
| SchaffersIllConditioned_20D | 0.0056 | 0.032 |
| LunacekBiR_5D | **1.0** | **0.9996** |
| LunacekBiR_10D | 0.2704 | **0.528** |
| LunacekBiR_20D | 0.0 | 0.0 |
| GG101me_5D | **0.9852** | **0.996** |
| GG101me_10D | **0.6604** | **0.8852** |
| GG101me_20D | 0.0608 | 0.2864 |
| GG21hi_5D | **0.88** | **0.7956** |
| GG21hi_10D | 0.242 | 0.488 |
| GG21hi_20D | 0.0088 | 0.0872 |
| **Total p(PPO mf wDeltaFitPop < iDE)** | 32.6% (15/46) | **54.3**% (25/46) |