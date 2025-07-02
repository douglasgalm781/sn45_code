import asyncio
from typing import List

from coding.protocol import LogicSynapse
from coding.schemas.tracking import TrackingInfo
from coding.utils.uids import get_miner_uids, get_hotkey_from_uid
import asyncio
import threading
from coding.constants import BLACKLISTED_COLDKEYS

def run_async_in_thread(coro):
    """
    Runs an async coroutine in a separate thread and returns its result.
    """
    result_container = []
    exception_container = []

    def target():
        try:
            result = asyncio.run(coro)
            result_container.append(result)
        except Exception as e:
            exception_container.append(e)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()

    if exception_container:
        raise exception_container[0]
    return result_container[0]


def gather_all_logics(validator) -> List[TrackingInfo]:
    uids = get_miner_uids(validator)
    uids = [uid for uid in uids if validator.metagraph.coldkeys[uid] not in BLACKLISTED_COLDKEYS]
    axons = [validator.metagraph.axons[uid] for uid in uids]
    synapse = LogicSynapse()

    # Run the async query in a separate thread
    responses = run_async_in_thread(
        validator.dendrite.aquery(
            axons=axons, synapse=synapse, timeout=45, deserialize=False
        )
    )
    return [
        TrackingInfo(
            logic=synapse.logic,
            block=validator.metagraph.block,
            hotkey=get_hotkey_from_uid(validator, uids[i]),
            uid=uids[i],
            score=0.0,
            score_timestamps=[],
        )
        for i, synapse in enumerate(responses)
    ]


def regrab_tracker(tracker: TrackingInfo, validator) -> TrackingInfo:
    uid = tracker.uid
    if tracker.hotkey != get_hotkey_from_uid(validator, uid):
        return tracker
    synapse = LogicSynapse()
    logic_synapse = validator.dendrite.query(
        axons=[validator.metagraph.axons[uid]],
        synapse=synapse,
        timeout=45,
        deserialize=False,
    )[0]
    return TrackingInfo(
        logic=logic_synapse.logic,
        block=validator.metagraph.block,
        hotkey=get_hotkey_from_uid(validator, uid),
        uid=uid,
        score=0.0,
    )
