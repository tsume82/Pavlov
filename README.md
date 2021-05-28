<p align="center">
  <h3 align="center">Pavlov</h3>
  <p align="center">
    A reinforcement learning python library interfacing with Kimeme
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Pavlov is a reinforcement python learning library providing the tools to optimize Kimeme RL scheduler and RL memes, automatically managing the training phase via KimemeCLI and providing KimemeGUI/KimemeCLI the information needed to test the trained models. The testing phase will be later integrated Kimeme-side, without the need for python RPC.
**This library is still in development and many functionalities are not available yet**


### Built With

The core libraries used by Pavlov are:
* [Ray RLlib](https://docs.ray.io/en/master/rllib.html)
* [Tensorflow 2.x](https://www.tensorflow.org/)
* [OpenAI GYM](https://gym.openai.com/)

Additional RL models can be wrapped in Pavlov via the ``Agent`` interface, from libraries such as
* [TensorForce](https://tensorforce.readthedocs.io/)
* [TF_Agents](https://www.tensorflow.org/agents)



<!-- GETTING STARTED -->
## Getting Started
Pavlov interface is modelled after TensorFlow one, providing powerful commands to train and test the model configured. The main interface methods are still WIP.

### Prerequisites
Currently tested with Python 3.6.9, not intended to work with Pytohn 2.x.

### Installation
Create a dedicated virtualenv using the ``requirements.txt`` file (TODO), eventually adding the libraries needed for custom environments or custom metrics

<!-- USAGE EXAMPLES -->
## Usage
TODO

<!-- ROADMAP -->
## Roadmap
See the internal KIMEMERL shared Google sheet (Gantt sheet)
