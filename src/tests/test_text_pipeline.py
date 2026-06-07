import json
from pathlib import Path

from torch.utils.data import DataLoader

from crest.cli_prepare_text import ByteTokenizer, build_episodes, write_split
from crest.config import CRESTConfig, DataConfig
from crest.data import ArrowEpisodicDataset, JsonlEpisodicDataset, collate_episodes
from crest.eval import evaluate
from crest.model import CRESTModel


def test_prepare_text_builds_shifted_episodic_arrow(tmp_path):
    tok = ByteTokenizer()
    episodes = build_episodes(["hello world"], tok, episode_steps=2, step_length=4, stride_tokens=0)
    assert episodes
    first = episodes[0]["steps"][0]
    assert first["labels"][:-1] == first["input_ids"][1:]
    write_split(episodes, tmp_path, eval_fraction=0.5, seed=1)
    assert (tmp_path / "train").exists()
    assert (tmp_path / "eval").exists()
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


def test_wikitext103_downloader_and_mock_training(tmp_path):
    import sys
    from unittest.mock import patch
    from crest.downloader import download_wikitext103
    from crest.cli_prepare_text import main as prepare_main
    from crest.train import run_training
    from crest.config import CRESTConfig, DataConfig, TrainingConfig

    # 1. Test downloader mock mode
    mock_raw_dir = tmp_path / "raw_wikitext103"
    download_path = download_wikitext103(dest_dir=mock_raw_dir, mock=True)
    assert download_path.exists()
    assert (download_path / "wiki.train.tokens").exists()
    assert (download_path / "wiki.valid.tokens").exists()

    # 2. Test prep pipeline with mock download via command arguments
    out_dir = tmp_path / "wikitext103_episodic"
    
    test_args = [
        "cli_prepare_text",
        "--out", str(out_dir),
        "--mock-download",
        "--episode-steps", "4",
        "--step-length", "16",
        "--tokenizer", "byte",
        "--eval-fraction", "0.2"
    ]
    with patch.object(sys, "argv", test_args):
        prepare_main()

    assert (out_dir / "train").exists()
    assert (out_dir / "eval").exists()
    assert (out_dir / "metadata.json").exists()

    with open(out_dir / "metadata.json", "r") as f:
        meta = json.load(f)
    assert meta["train"] > 0
    assert meta["eval"] > 0

    # 3. Verify mock training receives and processes it
    model_cfg = CRESTConfig(vocab_size=260, max_seq_len=16, max_steps=4, n_layers=1, d_model=32, n_heads=4, d_ffn=64, memory_slots=4)
    data_cfg = DataConfig(
        suite="wikitext103_episodic",
        task="arrow_episodic",
        vocab_size=260,
        episode_steps=4,
        step_length=16,
        path=str(out_dir),
        train_episodes=meta["train"],
        eval_episodes=meta["eval"]
    )
    assert ArrowEpisodicDataset(data_cfg, "train")[0].input_ids.shape == (4, 16)
    train_cfg = TrainingConfig(
        batch_size=2,
        max_steps=2,
        tbptt_k=2,
        output_dir=str(tmp_path / "run_out"),
        run_name="mock_wt103_run",
        log_every=1,
        eval_every=1,
        save_every=1
    )

    result = run_training(model_cfg, data_cfg, train_cfg)
    assert result["step"] == 2
    assert (tmp_path / "run_out" / "checkpoint_final.pt").exists()
