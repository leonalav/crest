import json
from pathlib import Path

from torch.utils.data import DataLoader

from crest.cli_prepare_text import ByteTokenizer, build_episodes, write_split
from crest.config import CRESTConfig, DataConfig
from crest.data import JsonlEpisodicDataset, collate_episodes
from crest.eval import evaluate
from crest.model import CRESTModel


def test_prepare_text_builds_shifted_episodic_jsonl(tmp_path):
    tok = ByteTokenizer()
    episodes = build_episodes(["hello world"], tok, episode_steps=2, step_length=4, stride_tokens=0)
    assert episodes
    first = episodes[0]["steps"][0]
    assert first["labels"][:-1] == first["input_ids"][1:]
    write_split(episodes, tmp_path, eval_fraction=0.5, seed=1)
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "eval.jsonl").exists()
    assert json.loads((tmp_path / "metadata.json").read_text())["episodes"] == len(episodes)


def test_jsonl_dataset_and_boundary_metrics(tmp_path):
    row = {
        "steps": [
            {"input_ids": [1, 2, 3, 4], "labels": [2, 3, 4, 5]},
            {"input_ids": [5, 6, 7, 8], "labels": [6, 7, 8, 9]},
        ]
    }
    for split in ["train", "eval"]:
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    cfg = DataConfig(task="jsonl_episodic", path=str(tmp_path), episode_steps=2, step_length=4, vocab_size=32)
    ds = JsonlEpisodicDataset(cfg, "eval")
    batch = collate_episodes([ds[0]])
    assert batch.input_ids.shape == (1, 2, 4)
    model = CRESTModel(CRESTConfig(vocab_size=32, max_seq_len=4, max_steps=4, n_layers=1, d_model=32, n_heads=4, d_ffn=64, memory_slots=4))
    metrics = evaluate(model, DataLoader(ds, batch_size=1, collate_fn=collate_episodes), max_batches=1)
    assert "boundary_loss" in metrics
    assert "boundary_accuracy" in metrics
