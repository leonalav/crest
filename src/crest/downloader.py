from __future__ import annotations

import os
from pathlib import Path

def download_wikitext103(dest_dir: Path | str | None = None, mock: bool = False) -> Path:
    """Download the WikiText-103 dataset using kagglehub.
    
    If mock is True, it will generate a lightweight mock wikitext dataset 
    with dummy token files in the dest_dir to allow fast testing without 
    performing a full 500MB download.
    """
    if mock:
        if dest_dir is None:
            raise ValueError("dest_dir must be provided when mock=True")
        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # Write dummy/mock files
        mock_train = " = Mock Train Title = \n\n This is a mock train paragraph for WikiText-103 mock training.\n"
        mock_valid = " = Mock Valid Title = \n\n This is a mock validation paragraph for WikiText-103.\n"
        mock_test = " = Mock Test Title = \n\n This is a mock test paragraph for WikiText-103.\n"
        
        (dest_path / "wiki.train.tokens").write_text(mock_train * 50, encoding="utf-8")
        (dest_path / "wiki.valid.tokens").write_text(mock_valid * 10, encoding="utf-8")
        (dest_path / "wiki.test.tokens").write_text(mock_test * 10, encoding="utf-8")
        (dest_path / "LICENSE.txt").write_text("Mock License", encoding="utf-8")
        (dest_path / "README.txt").write_text("Mock README", encoding="utf-8")
        return dest_path

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "The 'kagglehub' package is required to download WikiText-103 from Kaggle.\n"
            "Please install it using: pip install kagglehub"
        ) from exc

    print("[downloader] downloading vadimkurochkin/wikitext-103 dataset via kagglehub...")
    path_str = kagglehub.dataset_download("vadimkurochkin/wikitext-103")
    return Path(path_str)
