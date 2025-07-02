# FAQ


## How do I determine how good my miner is?

Check wandb - https://wandb.ai/gen42/gen42. Complete the miner-average-score.ipynb notebook in /notebooks.

## How do I know if my miner is working?

Ensure you can curl it: `curl <miner-ip>:<miner-port>`.

Ensure that you are seeing logs like "Received query" in your pm2 logs.

Ensure that there is no errors in the logs, warnings are fine.

## What are these pydantic errors?

Just ignore them.

## How can i remove debug logging?

Edit `coding/utils/config.py` and remove line 301 `bt.debug()`.

## How can i disable trace logging?

Edit `coding/utils/config.py` and remove line 300 `bt.trace()`.

## How is scoring done?

The scoring depends on the task, however primarily it is done in the following route:

1. Get code from The Stack
2. Rewrite the code with an LLM to ensure that lookups are not possible
3. Grab a chunk from that code and erase it
4. Provide the remaining code to the miner
5. Compare the chunk to the miner's response using Cosine Similarity with CodeBERT
6. Return the score
