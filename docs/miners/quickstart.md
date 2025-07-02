# Quickstart to Mining

## Installation


This repository requires python3.9 or higher. To install it, simply clone this repository and run the [install.sh](./install.sh) script.
```bash
git clone https://github.com/brokespace/code
cd code
python -m pip install --use-deprecated=legacy-resolver -r requirements.txt
python -m pip install --use-deprecated=legacy-resolver -e .
python -m pip uninstall uvloop # b/c it causes issues with threading/loops
```


## OpenRouter API Key Setup

You need an OpenRouter Provisioning Key to run the miners. You can get one by going to the [OpenRouter website](https://openrouter.ai/settings/provisioning-keys) and creating one.

This key should be added to the `.env` file as `PROVISIONING_API_KEY`. An example `.env.example.miner` file is provided in the root of the repository.

This key will be used to provision temporary api keys for the miner to provide to the Validator so that it can use it to validate the miner's logic. 

The validator will request a key to be created, and after it is done with the evaluation, a request will be sent to have the key deleted.


## How to Run
You can use the following command to run a miner or a validator. 

```bash
python <SCRIPT_PATH>
    --netuid 45
    --subtensor.network <finney/local/test>
    --neuron.device cuda
    --wallet.name <your wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --axon.port # VERY IMPORTANT: set the port to be one of the open TCP ports on your machine
```

where `SCRIPT_PATH` is either: 
1. neurons/miner.py
2. neurons/validator.py

For ease of use, you can run the scripts as well with PM2. Installation of PM2 is: 
**On Linux**:
```bash
sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
``` 

Example of running an openai miner:

```bash
pm2 start neurons/miner.py --interpreter python3 --name miner -- --netuid XY  --subtensor.network finney --wallet.name coldkey --wallet.hotkey hotkey --neuron.model_id gpt4 --axon.port 8091 --logging.debug --miner.name openai
```

## Subnet Wallet Registration
Register your wallet on the subnet: 
```
btcli s register --subtensor.network finney --netuid 45
```

Testnet: 
```
btcli s register --subtensor.network test --netuid 171
```


# Testnet 
We highly recommend that you run your miners on testnet before deploying on main. This is give you an opportunity to debug your systems, and ensure that you will not lose valuable immunity time. The SN1 testnet is **netuid 171**. 

In order to run on testnet, you will need to go through the same hotkey registration proceure as on main, but using **testtao**. You will need to ask for some in the community discord if you do not have any. 

To run:

```bash
pm2 start neurons/miner.py --interpreter python3 --name miner -- --netuid 171  --subtensor.network test --wallet.name test_coldkey --wallet.hotkey test_hotkey --neuron.model_id gpt4 --axon.port 8091 --logging.debug --miner.name openai
```


# Ramping up


## Testing

A notebook is [provided](https://github.com/brokespace/code/blob/main/notebooks/sample-swe-task.ipynb)

Youre going to want:
- [.env](https://github.com/brokespace/code/blob/main/.env.example) setup correctly
- If using claude modify [swebase](https://github.com/brokespace/code/blob/main/notebooks/example_submission/swebase.py) to support claude. 
- Docker setup locally


Look at submission on https://www.swebench.com/ and build off one of those to perform well on the tasks. Our testing set is an EXACT copy of the swebench testing methodology, so anything that performs well there, will perform well on our subnet
