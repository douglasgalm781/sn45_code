import os
import requests
from uuid import uuid4
from coding.protocol import ProvisionKeySynapse

BASE_URL = "https://openrouter.ai/api/v1/keys"
def miner_process(self, synapse: ProvisionKeySynapse) -> ProvisionKeySynapse:
    """
    The miner process function is called every time the miner receives a request. This function should contain the main logic of the miner.
    """
    provision_key = os.getenv("PROVISIONING_API_KEY")
    
    if synapse.action == "create":
        response = requests.post(
            f"{BASE_URL}",
            headers={
                "Authorization": f"Bearer {provision_key}",
                "Content-Type": "application/json"
            },
            json={
                "name": f"mining-{str(uuid4())}",
                "label": "mining",
                #"limit": 1000  # Optional credit limit
            }
        )
        synapse.api_key = response.json()['key']
        synapse.key_hash = response.json()['data']['hash']
    elif synapse.action == "delete":
        response = requests.delete(
            f"{BASE_URL}/{synapse.key_hash}",
            headers={
                "Authorization": f"Bearer {provision_key}",
            }
        )
        synapse.api_key = None
        synapse.key_hash = None
    return synapse
