import torch
from torch.utils.data import DataLoader

from crest.config import CRESTConfig, DataConfig, TrainingConfig
from crest.data import SyntheticKeyValueDataset, collate_episodes
from crest.losses import dpo_loss, lm_loss
from crest.model import CRESTModel
from crest.train import build_optimizer, train_episode_batch


def test_synthetic_dataset_collates_episode_steps():
    ds = SyntheticKeyValueDataset(DataConfig(episode_steps=4, step_length=6, train_episodes=3), split="train")
    batch = collate_episodes([ds[0], ds[1]])
    assert batch.input_ids.shape == (2, 4, 6)
    assert batch.labels.shape == (2, 4, 6)
    assert batch.step_idx.shape == (2, 4)


def test_lm_loss_runs():
    logits = torch.randn(2, 3, 11, requires_grad=True)
    labels = torch.tensor([[1, -100, 3], [4, 5, 6]])
    loss = lm_loss(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)


def test_dpo_loss_prefers_higher_policy_ratio():
    good = dpo_loss(torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), beta=1.0)
    bad = dpo_loss(torch.tensor([0.0]), torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([0.0]), beta=1.0)
    assert good < bad


def test_train_episode_batch_smoke():
    cfg = CRESTConfig(vocab_size=256, max_seq_len=8, max_steps=8, n_layers=1, d_model=32, n_heads=4, d_ffn=64, memory_slots=4)
    model = CRESTModel(cfg)
    train_cfg = TrainingConfig(batch_size=2, tbptt_k=2, learning_rate=1e-3)
    opt = build_optimizer(model, train_cfg)
    data_cfg = DataConfig(vocab_size=256, num_keys=8, num_values=32, episode_steps=4, step_length=8, train_episodes=4)
    loader = DataLoader(SyntheticKeyValueDataset(data_cfg), batch_size=2, collate_fn=collate_episodes)
    batch = next(iter(loader))
    metrics = train_episode_batch(model, batch, opt, train_cfg)
    assert metrics["loss"] >= 0.0
    assert 0.0 <= metrics["gate_mean"] <= 1.0
