import torch
from torch.utils.data import DataLoader

from crest.ablation import memory_sweep_configs, no_state_config
from crest.alignment import sft_loss_for_batch, trajectory_logprob
from crest.baselines import FullAttentionBaseline
from crest.config import CRESTConfig, DataConfig, TrainingConfig
from crest.data import build_dataset, collate_episodes
from crest.eval import evaluate
from crest.metrics import component_parameter_counts, estimate_episode_flops
from crest.model import CRESTModel
from crest.report import model_report
from crest.train import run_training


def tiny_cfg():
    return CRESTConfig(vocab_size=256, max_seq_len=8, max_steps=8, n_layers=1, d_model=32, n_heads=4, d_ffn=64, memory_slots=4)


def tiny_data(task="multi_hop"):
    return DataConfig(vocab_size=256, num_keys=8, num_values=32, episode_steps=4, step_length=8, train_episodes=4, eval_episodes=4, task=task)


def test_multihop_and_tooltrace_builders():
    for task in ["multi_hop", "tool_trace"]:
        ds = build_dataset(tiny_data(task), "train")
        item = ds[0]
        assert item.input_ids.shape == (4, 8)
        assert item.labels.shape == (4, 8)


def test_eval_and_report_surfaces():
    cfg = tiny_cfg()
    model = CRESTModel(cfg)
    loader = DataLoader(build_dataset(tiny_data("multi_hop"), "eval"), batch_size=2, collate_fn=collate_episodes)
    metrics = evaluate(model, loader, max_batches=1)
    assert "recall_accuracy" in metrics
    report = model_report(model, cfg, episode_steps=4)
    assert report["total_parameters"] > 0
    assert estimate_episode_flops(cfg, 4)["total"] > 0
    assert sum(component_parameter_counts(model).values()) == model.count_parameters()


def test_ablation_configs_and_full_attention_baseline():
    cfg = tiny_cfg()
    no_state = no_state_config(cfg)
    assert not no_state.use_state_read and not no_state.use_state_write
    assert set(memory_sweep_configs(cfg, [2, 4])) == {"M2", "M4"}
    baseline = FullAttentionBaseline(cfg)
    logits = baseline(torch.randint(0, cfg.vocab_size, (2, 4, 8)))
    assert logits.shape == (2, 4, 8, cfg.vocab_size)


def test_alignment_sft_logprob_surfaces():
    cfg = tiny_cfg()
    model = CRESTModel(cfg)
    batch = collate_episodes([build_dataset(tiny_data("tool_trace"), "train")[0], build_dataset(tiny_data("tool_trace"), "train")[1]])
    loss = sft_loss_for_batch(model, batch)
    lp = trajectory_logprob(model, batch.input_ids, batch.labels, batch.step_idx)
    assert torch.isfinite(loss)
    assert lp.shape == (2,)


def test_run_training_tiny_harness(tmp_path):
    cfg = tiny_cfg()
    data_cfg = tiny_data("tool_trace")
    train_cfg = TrainingConfig(batch_size=2, max_steps=2, tbptt_k=2, output_dir=str(tmp_path), run_name="tiny", log_every=1, eval_every=1, save_every=1)
    result = run_training(cfg, data_cfg, train_cfg)
    assert result["step"] == 2
    assert (tmp_path / "checkpoint_final.pt").exists()
