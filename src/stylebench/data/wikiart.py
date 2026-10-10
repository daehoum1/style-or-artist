"""WikiArt (ArtGAN crawl, 81,444 images in 27 style folders) adapter."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from .base import DatasetAdapter, normalize_label


class WikiArtAdapter(DatasetAdapter):
    def required_files(self):
        return [str(Path(self.cfg["root"]) / self.cfg["label_file"])]

    def load_raw(self) -> pd.DataFrame:
        root = Path(self.cfg["root"])
        w = pd.read_csv(root / self.cfg["label_file"])
        folder = w["file"].str.split("/").str[0]
        fname = w["file"].str.split("/").str[1]
        # style: folder name; verify against the integer style id (must agree exactly)
        folders = sorted(folder.unique())
        sid = folder.map({f: 140 + i for i, f in enumerate(folders)})
        if not (sid == w["style"]).all():
            raise ValueError("WikiArt style ids disagree with folder order")
        gnames = self.cfg["genre_names"]
        genre = (w["genre"] - self.cfg["genre_id_offset"]).map(lambda i: gnames[i])
        artist = fname.str.split("_").str[0]  # crawl slug, e.g. 'claude-monet'
        paths = self._resolve_paths(root, w["file"])
        df = pd.DataFrame({
            "image_id": w["file"],
            "path": paths,
            "style": folder.map(normalize_label),
            "artist": artist.map(normalize_label),
            "genre": genre.map(normalize_label),
            "raw_style_folder": folder,
            "raw_artist_slug": artist,
            "raw_artist_id": w["artist"],
            "raw_genre_id": w["genre"],
            "raw_style_id": w["style"],
        })
        return df

    @staticmethod
    def _resolve_paths(root: Path, files: pd.Series) -> list:
        """The label file and the disk use different (both mojibake) encodings for ~770 non-ASCII names.
        Unmatched names are resolved by an ASCII-only key within the same folder, accepted only if the
        key is unique on both sides; anything else stays unresolved (path=None, recorded by cleaning)."""
        def key(n):
            return re.sub(r"[^a-z0-9._-]", "", n.lower())
        out = []
        disk_cache: dict = {}
        for f in files:
            if (root / f).exists():
                out.append(str(root / f))
                continue
            folder, name = f.split("/", 1)
            if folder not in disk_cache:
                names = [n for n in os.listdir(root / folder)]
                ks: dict = {}
                for n in names:
                    ks.setdefault(key(n), []).append(n)
                disk_cache[folder] = ks
            cand = disk_cache[folder].get(key(name), [])
            out.append(str(root / folder / cand[0]) if len(cand) == 1 else None)
        # uniqueness of resolved paths is asserted by the caller (cleaning)
        return out
