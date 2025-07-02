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

import os
import copy
import wandb
import coding
import logging
import bittensor as bt
from logging.handlers import RotatingFileHandler


EVENTS_LEVEL_NUM = 38
DEFAULT_LOG_BACKUP_COUNT = 10


def setup_events_logger(full_path, events_retention_size):
    logging.addLevelName(EVENTS_LEVEL_NUM, "EVENT")

    logger = logging.getLogger("event")
    logger.setLevel(EVENTS_LEVEL_NUM)

    def event(self, message, *args, **kws):
        if self.isEnabledFor(EVENTS_LEVEL_NUM):
            self._log(EVENTS_LEVEL_NUM, message, args, **kws)

    logging.Logger.event = event

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        os.path.join(full_path, "events.log"),
        maxBytes=events_retention_size,
        backupCount=DEFAULT_LOG_BACKUP_COUNT,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(EVENTS_LEVEL_NUM)
    logger.addHandler(file_handler)

    return logger


def should_reinit_wandb(self):
    # Check if wandb run needs to be rolled over.
    return (
        not self.config.wandb.off
        and self.step
        and self.step % self.config.wandb.run_step_length == 0
    )


def init_wandb(self, reinit=False):
    """Starts a new wandb run."""
    uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
    spec_version = str(coding.__spec_version__)
    tags = [
        self.wallet.hotkey.ss58_address,
        coding.__version__,
        str(coding.__spec_version__),
        f"netuid_{self.metagraph.netuid}",
    ]

    if self.config.mock:
        tags.append("mock")
    # for task in self.active_tasks:
    # tags.append(task)
    if self.config.neuron.disable_set_weights:
        tags.append("disable_set_weights")

    wandb_config = {
        key: copy.deepcopy(self.config.get(key, None))
        for key in ("neuron", "reward", "netuid", "wandb")
    }
    wandb_config["neuron"].pop("full_path", None)

    self.wandb = wandb.init(
        anonymous="allow",
        reinit=reinit,
        project=(
            self.config.wandb.project_name
            if self.config.netuid == 45
            else self.config.wandb.project_name + "testnet"
        ),
        entity=self.config.wandb.entity,
        config=wandb_config,
        mode="offline" if self.config.wandb.offline else "online",
        dir=self.config.neuron.full_path,
        tags=tags,
        notes=self.config.wandb.notes,
        name=f"{uid}-{spec_version}",
    )
    bt.logging.success(f"Started a new wandb run <blue> {self.wandb.name} </blue>")


def reinit_wandb(self):
    """Reinitializes wandb, rolling over the run."""
    self.wandb.finish()
    init_wandb(self, reinit=True)


def clean_wandb(self):
    """Cleans wandb, deleting all runs."""
    try:
        if not self.wandb:
            wandb_dir = self.config.neuron.full_path + "/wandb"
            if os.path.exists(wandb_dir):
                for root, dirs, files in os.walk(wandb_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(wandb_dir)
        else:
            self.wandb.finish()
            wandb.delete_all()
    except Exception as e:
        bt.logging.error(f"Error cleaning wandb: {e}")


def init_wandb_if_not_exists(self):
    if self.config.netuid != 45 and self.config.netuid != 171:
        return
    if not self.config.wandb.on:
        return
    if getattr(self, "wandb", None):
        return
    init_wandb(self)


def log_event(self, event):
    if self.config.netuid != 45 and self.config.netuid != 171:
        return

    if not self.config.wandb.on:
        return

    if not getattr(self, "wandb", None):
        init_wandb(self)
    try:
        # Log the event to wandb.
        self.wandb.log(event)
    except Exception as e:
        bt.logging.error(f"Error logging event: {e}")
        try:
            init_wandb(self)
            self.wandb.log(event)
        except Exception as e:
            bt.logging.error(f"Error logging event: {e}")
