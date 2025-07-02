# Sample Miners


## Qwen Mistral Miner

To get started on this miner you are going to want two models, `CodeQwen` and `Mistral`. This setup requires quite a bit of VRAM, I would suggest at a minimum 24gb of VRAM.



### Starting LLM's

Either use Python or Docker to start the LLMs. If using Docker you will need to get the [cuda container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.13.5/install-guide.html).

#### Using Python
Create a venv for VLLM, this venv must not be the same one you use to run the miner

```bash
python -m venv vllm
source vllm/bin/activate
pip install vllm
```

Then start the LLM's

```bash
pm2 start --name "mistral" "vllm serve thesven/Mistral-7B-Instruct-v0.3-GPTQ  --max-model-len 4096 --quantization gptq --dtype half --gpu-memory-utilization 0.40 --port 8001"
```

```bash
pm2 start --name "qwen" "vllm serve Qwen/CodeQwen1.5-7B-AWQ  --max-model-len 4096 --quantization awq --dtype half --gpu-memory-utilization 0.40 --port 8000"
```

#### Using Docker

The commands below will run VLLM on device=0 (gpu0), be sure to modify that if you want to run on a different gpu

```bash
sudo docker run -d -p 8000:8000 --gpus device=0 --ipc host --name codeqwen docker.io/vllm/vllm-openai:latest --model Qwen/CodeQwen1.5-7B-AWQ --max-model-len 8096 --dtype half  --gpu-memory-utilization 0.4
```

```bash
sudo docker run -d -p 8001:8001  --gpus device=0 --ipc host --name mistral-instruct docker.io/vllm/vllm-openai:latest --model thesven/Mistral-7B-Instruct-v0.3-GPTQ --max-model-len 8912  --dtype half --gpu-memory-utilization 0.40
```


### Starting the Miner

Exit the previous venv for vllm, either creating a new venv or using your default python interpreter.

```
pm2 start neurons/miner.py --interpreter python3 --name miner -- --netuid 45 --subtensor.network finney --wallet.name coldkey --wallet.hotkey hotkey --neuron.model_id Qwen/CodeQwen1.5-7B-AWQ --axon.port 8091 --logging.debug --miner.name qwen_mistral
```