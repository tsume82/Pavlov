# DE gaussian PPO mf wDeltaFitPop vs jDE

| Function    | p(PPO mf wDeltaFitPop < jDE) with AUC metric | p(PPO mf wDeltaFitPop < jDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.3412 | 0.364 |
| Discus | 0.4812 | 0.3632 |
| Ellipsoid | 0.2728 | 0.354 |
| Katsuura | **0.59** | **0.54** |
| Rastrigin | 0.348 | 0.4348 |
| Rosenbrock | 0.3484 | 0.2896 |
| Schaffers | 0.344 | 0.3608 |
| Schwefel | 0.2404 | 0.276 |
| Sphere | 0.1856 | 0.3248 |
| Weierstrass | **0.6032** | **0.7032** |
| AttractiveSector_5D | **0.5592** | **0.9768** |
| AttractiveSector_10D | 0.404 | 0.4176 |
| AttractiveSector_20D | 0.0 | 0.0032 |
| BuecheRastrigin_5D | **1.0** | **0.9608** |
| BuecheRastrigin_10D | 0.2832 | 0.4812 |
| BuecheRastrigin_20D | 0.0 | 0.0004 |
| CompositeGR_5D | **0.9744** | **0.9356** |
| CompositeGR_10D | 0.3336 | 0.4428 |
| CompositeGR_20D | 0.0072 | 0.0064 |
| DifferentPowers_5D | **0.9748** | **0.9304** |
| DifferentPowers_10D | 0.252 | 0.4456 |
| DifferentPowers_20D | 0.0016 | 0.0192 |
| LinearSlope_5D | **0.968** | **0.9228** |
| LinearSlope_10D | 0.3772 | **0.5052** |
| LinearSlope_20D | 0.0 | 0.0 |
| SharpRidge_5D | **0.9784** | **0.9204** |
| SharpRidge_10D | 0.3992 | **0.5388** |
| SharpRidge_20D | 0.0 | 0.0024 |
| StepEllipsoidal_5D | **0.9792** | **0.9204** |
| StepEllipsoidal_10D | 0.376 | **0.612** |
| StepEllipsoidal_20D | 0.004 | 0.0504 |
| RosenbrockRotated_5D | **0.992** | **0.9072** |
| RosenbrockRotated_10D | 0.3396 | 0.468 |
| RosenbrockRotated_20D | 0.0064 | 0.0 |
| SchaffersIllConditioned_5D | **0.8764** | **0.8236** |
| SchaffersIllConditioned_10D | 0.3616 | 0.4256 |
| SchaffersIllConditioned_20D | 0.0064 | 0.0072 |
| LunacekBiR_5D | **1.0** | **0.9928** |
| LunacekBiR_10D | 0.3688 | 0.3848 |
| LunacekBiR_20D | 0.0 | 0.0 |
| GG101me_5D | **0.9536** | **0.7352** |
| GG101me_10D | 0.4036 | 0.4788 |
| GG101me_20D | 0.0228 | 0.09 |
| GG21hi_5D | **0.8344** | **0.7952** |
| GG21hi_10D | 0.3956 | 0.4376 |
| GG21hi_20D | 0.0028 | 0.0308 |
| **Total p(PPO mf wDeltaFitPop < jDE)** | 30.4% (14/46) | 37.0% (17/46) |

# DE gaussian PPO mf wDeltaFitPop vs iDE

| Function    | p(PPO mf wDeltaFitPop < iDE) with AUC metric | p(PPO mf wDeltaFitPop < iDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.37 | 0.4032 |
| Discus | 0.4444 | **0.6472** |
| Ellipsoid | 0.3136 | **0.5696** |
| Katsuura | 0.464 | 0.4504 |
| Rastrigin | 0.3576 | **0.514** |
| Rosenbrock | 0.2876 | 0.404 |
| Schaffers | 0.4048 | 0.4652 |
| Schwefel | 0.266 | **0.5764** |
| Sphere | 0.3244 | 0.4552 |
| Weierstrass | **0.6516** | **0.7016** |
| AttractiveSector_5D | **0.5168** | **0.9936** |
| AttractiveSector_10D | **0.5024** | **0.6196** |
| AttractiveSector_20D | 0.0 | 0.0012 |
| BuecheRastrigin_5D | **1.0** | **1.0** |
| BuecheRastrigin_10D | 0.3084 | 0.3696 |
| BuecheRastrigin_20D | 0.0048 | 0.1216 |
| CompositeGR_5D | **0.9808** | **0.9688** |
| CompositeGR_10D | 0.2196 | 0.3888 |
| CompositeGR_20D | 0.0836 | 0.0376 |
| DifferentPowers_5D | **0.9968** | **0.9676** |
| DifferentPowers_10D | 0.3008 | **0.6712** |
| DifferentPowers_20D | 0.0008 | 0.0008 |
| LinearSlope_5D | **0.9632** | **0.8704** |
| LinearSlope_10D | 0.2404 | 0.1068 |
| LinearSlope_20D | 0.0 | 0.0012 |
| SharpRidge_5D | **0.9964** | **0.92** |
| SharpRidge_10D | 0.2012 | 0.3048 |
| SharpRidge_20D | 0.0 | 0.0 |
| StepEllipsoidal_5D | **0.9868** | **0.7424** |
| StepEllipsoidal_10D | 0.4196 | **0.7464** |
| StepEllipsoidal_20D | 0.0 | 0.0016 |
| RosenbrockRotated_5D | **1.0** | **0.9864** |
| RosenbrockRotated_10D | **0.7892** | **0.672** |
| RosenbrockRotated_20D | 0.018 | 0.0012 |
| SchaffersIllConditioned_5D | **0.9236** | **0.9544** |
| SchaffersIllConditioned_10D | 0.2708 | 0.2844 |
| SchaffersIllConditioned_20D | 0.0008 | 0.0032 |
| LunacekBiR_5D | **1.0** | **0.9912** |
| LunacekBiR_10D | 0.3548 | 0.4972 |
| LunacekBiR_20D | 0.0 | 0.0 |
| GG101me_5D | **0.9824** | **0.9888** |
| GG101me_10D | **0.684** | **0.8928** |
| GG101me_20D | 0.04 | 0.1144 |
| GG21hi_5D | **0.8344** | **0.7028** |
| GG21hi_10D | 0.2652 | 0.3564 |
| GG21hi_20D | 0.0028 | 0.0184 |
| **Total p(PPO mf wDeltaFitPop < iDE)** | 34.8% (16/46) | 47.8% (22/46) |