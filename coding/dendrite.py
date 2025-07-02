# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2024 Macrocosmos
# Copyright © 2024 Brokespace


# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt
from typing import List, Any


class DendriteResponseEvent:
    def __init__(
        self, responses: List[bt.Synapse], uids, timeout: float, axons: List[Any]
    ):
        self.uids = uids
        self.completions = []
        self.status_messages = []
        self.status_codes = []
        self.timings = []
        self.hotkeys = []
        self.axons = axons
        for synapse in responses:
            self.completions.append(synapse.completion)
            self.status_messages.append(synapse.dendrite.status_message)

            if len(synapse.completion) == 0 and synapse.dendrite.status_code == 200:
                synapse.dendrite.status_code = 204

            self.status_codes.append(synapse.dendrite.status_code)

            if (synapse.dendrite.process_time) and (
                synapse.dendrite.status_code == 200
                or synapse.dendrite.status_code == 204
            ):
                self.timings.append(synapse.dendrite.process_time)
            elif synapse.dendrite.status_code == 408:
                self.timings.append(timeout)
            else:
                self.timings.append(0)  # situation where miner is not alive

        self.completions = [synapse.completion for synapse in responses]
        self.timings = [
            synapse.dendrite.process_time or timeout for synapse in responses
        ]
        self.status_messages = [
            synapse.dendrite.status_message for synapse in responses
        ]
        self.status_codes = [synapse.dendrite.status_code for synapse in responses]

        self.miner_hotkeys = [axon.hotkey for axon in axons]

    def __state_dict__(self):
        return {
            "uids": self.uids.tolist(),
            "completions": self.completions,
            "timings": self.timings,
            "status_messages": self.status_messages,
            "status_codes": self.status_codes,
            "miner_hotkeys": self.miner_hotkeys,
        }

    def __repr__(self):
        return f"DendriteResponseEvent(uids={self.uids}, completions={self.completions}, timings={self.timings}, status_messages={self.status_messages}, status_codes={self.status_codes}, miner_hotkeys={self.hotkeys})"
