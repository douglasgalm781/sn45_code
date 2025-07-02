import bittensor as bt

from coding.protocol import ProvisionKeySynapse
from coding.utils.uids import get_uid_from_hotkey
from coding.finetune.tracker import run_async_in_thread

class APIKey:
    def __init__(self, hotkey: str, validator):
        self.hotkey = hotkey
        self.validator = validator
        self.wallet = validator.wallet
        self.key_synapse = ProvisionKeySynapse()

    @property
    def key(self):
        if self.key_synapse.api_key:
            return self.key_synapse.api_key
        dendrite = bt.dendrite(wallet=self.wallet)
        uid = get_uid_from_hotkey(self.validator, self.hotkey)
        responses = run_async_in_thread(
            dendrite.aquery(
                axons=[self.validator.metagraph.axons[uid]],
                synapse=ProvisionKeySynapse(action="create"),
                timeout=20,
                deserialize=False,
            )
        )
        if len(responses) == 0:
            return None
        if len(responses) > 1:
            raise Exception(f"Expected 1 response, got {len(responses)}")
        self.key_synapse = responses[0]
        return self.key_synapse.api_key

    def delete(self):
        dendrite = bt.dendrite(wallet=self.wallet)
        uid = get_uid_from_hotkey(self.validator, self.hotkey)
        responses = run_async_in_thread(
            dendrite.aquery(
                axons=[self.validator.metagraph.axons[uid]],
                synapse=ProvisionKeySynapse(
                    action="delete", key_hash=self.key_synapse.key_hash
                ),
                timeout=20,
                deserialize=False,
            )
        )
        print(responses)
