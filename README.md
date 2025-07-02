# **SWE-Rizzo - Software Engineering on Bittensor - A Team Rizzo Subnet** <!-- omit in toc -->

<!-- ### Decentralizing Code Generation  -->

<!-- [Discord](https://discord.gg/code) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper) -->

<!-- </div> -->

<!-- --- -->

# Introduction

Gen42 leverages the Bittensor network to provide decentralized software engineering services. Our focus is on creating robust, scalable tools for software engineering, powered by open-source large language models. These tools should benefit both Bittensor community and the broader software engineering ecosystem. To achieve this, our miners compete to develop the best SWE (software engineering) pipelines and we score their performance.

As a Team Rizzo subnet, there are three pillars we aim to solve:
1. A hard real world problem, benchmarked by a leaderboard against the world's best.
2. A focus on enabling downstream products. This subnet should lead to many products.
3. A focus on integration - Other subnets in the Bittensor ecosystem should benefit from this work, leading to a stronger Bittensor.

## Problem Statement
We aim to solve a critical real world problem. Despite millions of dollars of investment over decades, even the most robust deployed software has bugs. We aim to develop tools that can help automatically fix issues within the software engineering process, from code generation to testing, deployment, and maintenance. This will not only save money and labor time, but also lead to software that is more performant and safer. 

In this subnet, miners are incentivized to develop SWE pipelines that are able to complete tasks from the SWE-Bench dataset. These pipelines are then evaluated against Princeton's [SWE-Bench](https://www.swebench.com/) to determine the performance of the miner. Additionally, the winning pipelines get published to the [SWE-Bench Leaderboard](https://www.swebench.com).


## Products

Below are some links to a functional, subscription based product that has been built on previous incarnations of the subnet. We expect to build several additional products and to work lessons learned to refine our incentive mechanism.

:link:**Links to existing products described below:** <br>

- [Gen42 Home](https://www.gen42.ai)
- [Gen42 Chat](https://chat.gen42.ai)
- [Gen42 API](http://api.gen42.ai)


#### Chat App

We provide a chat frontend that allows users to interact with our subnet. The primary offering of this app is code-based QnA.

#### Code Completion
<!-- 
Code completion has exploded in recent years, tools like [Github Copilot](https://github.com/features/copilot) are extremely popular but lack in some manners.  -->

<!-- Our subnet aims to compete with Copilot by offering code completion hosted on Bittensor through [Continue.dev](https://continue.dev/). Unlike Copilot we will not be relying on OpenAI. Our miners will be running open-source code-focused LLMs which have proven to be faster and smarter than the product Copilot uses (GPT Codex).  -->

<!-- With an unoptimized miner we have already found that  -->

We provide an openai compliant api capable of being utilized with [continue.dev](https://continue.dev/). For information on getting started visit [Gen42](https://www.gen42.ai). 

---


#### SWE CLI

We provide a CLI tool that allows users to take advantage of the SWE pipelines our miners have developed. We expect this to be used by other subnets for a stronger Bittensor.

## Mining and Validating

### Validators

To get started as a validator, follow the [Validator Quickstart Guide](./docs/validators/quickstart.md).


### Miners

To begin mining, refer to the [Miner Quickstart Guide](./docs/miners/quickstart.md).

Miners are to develop SWE-Bench pipelines that will be evaluated. They must host the code for their pipeline in a synapse to allow for validators to access it. 

The code for their pipeline must be a python script that takes in a repository and a git issue and returns a completed patch for the issue. 

### Incentive Mechanism

The incentive mechanism for miners is as follows:

- A task is grabbed from the SWE-Bench dataset.
- A miner's code is grabbed from their synapse.
- The miner's code is transferred into a Docker container with the repository and issue, where it is then executed.
- The provided patch is then evaluated using the SWE-Bench eval method.
- The miner is then rewarded based on the performance of the patch.


##### Disclaimer

This repo is a fork of Subnet 1, [Prompting](https://github.com/macrocosm-os/prompting/tree/main). Credit for the amazing code goes to them, they did a wonderful job.
