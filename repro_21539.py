"""Repro for the val/test/predict frame-skipping in Lightning-AI/pytorch-lightning#21539.

Runs one short fit with a fast validation loop under each candidate _update_n,
capturing what tqdm actually writes to the stream, and reports how many frames
were drawn and where the bar was when it closed.

    python repro_21539.py

No GPU, no network. ~10 seconds.
"""

import io
import logging
import re
import time
import warnings
from contextlib import redirect_stdout

import torch
from torch.utils.data import DataLoader, Dataset

import lightning as L
from lightning.pytorch.callbacks.progress import tqdm_progress

VAL_BATCHES = 100  # 400 samples / batch_size 4

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def master(bar, value):
    """What main does today: assign n, redraw unconditionally."""
    if not bar.disable:
        bar.n = value
        bar.refresh()


def update_only(bar, value):
    """This PR as originally submitted."""
    if not bar.disable:
        bar.update(value - bar.n)


def update_then_refresh(bar, value):
    """This PR after 5f98efc."""
    if not bar.disable:
        bar.update(value - bar.n)
        bar.refresh()


class Rand(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.randn(32)


class Boring(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def training_step(self, batch, batch_idx):
        return self.layer(batch).sum()

    def validation_step(self, batch, batch_idx):
        # ~500 it/s: faster than tqdm's 0.1s mininterval, which is the
        # condition under which update() defers a redraw.
        time.sleep(0.002)
        return self.layer(batch).sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def run(update_n):
    tqdm_progress._update_n = update_n
    stream = io.StringIO()
    with redirect_stdout(stream):
        L.Trainer(
            max_epochs=1,
            logger=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            accelerator="cpu",
            num_sanity_val_steps=0,
        ).fit(
            Boring(),
            DataLoader(Rand(40), batch_size=4),
            DataLoader(Rand(VAL_BATCHES * 4), batch_size=4),
        )

    # tqdm redraws by rewriting the line after a \r, so split on it.
    frames = [f for f in stream.getvalue().split("\r") if "Validation DataLoader" in f]
    positions = [int(m.group(1)) for f in frames if (m := re.search(rf"(\d+)/{VAL_BATCHES}", f))]
    return len(positions), (positions[-1] if positions else None)


print(f"validation bar, {VAL_BATCHES} batches\n")
print(f"{'_update_n':<34} {'frames drawn':>12} {'final position':>15}")
print("-" * 63)
for name, fn in [
    ("bar.n = value; bar.refresh()", master),
    ("bar.update(value - bar.n)", update_only),
    ("bar.update(...); bar.refresh()", update_then_refresh),
]:
    drawn, final = run(fn)
    flag = "" if final == VAL_BATCHES else "   <-- closed before finishing"
    print(f"{name:<34} {drawn:>12} {str(final) + '/' + str(VAL_BATCHES):>15}{flag}")
