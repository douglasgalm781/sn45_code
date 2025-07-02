from coding.protocol import HFModelSynapse


def miner_process(self, synapse: HFModelSynapse) -> HFModelSynapse:
    """
    The miner process function is called every time the miner receives a request. This function should contain the main logic of the miner.
    """
    synapse.model_name = "microsoft/Phi-3-mini-128k-instruct"
    synapse.competition_id = 1

    return synapse
