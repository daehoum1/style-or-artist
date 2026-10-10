# Datasets

Images are **not** redistributed in this repository. Obtain them from the original sources and place
(or symlink) them under `data/`, or edit `root:` in `configs/datasets/*.yaml`.

| dataset | expected location | contents required | labels used |
|---|---|---|---|
| WikiArt | `data/wikiart/` | the WikiArt image collection packaged for ArtGAN (27 style folders), see https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset, plus the label file `wclasses.csv` (columns `file,artist,genre,style` with ArtGAN integer ids) | style (folder), artist (file-name slug), genre (`wclasses.csv` id; 139 = missing) |
| MultitaskPainting100k | `data/multitask_painting/` | `images/` and the original ground-truth files `MultitaskPainting100k_Dataset_groundtruth/groundtruth_multiloss_{train,test}_header.csv` (Bianco et al., *Expert Systems with Applications*, 2019) | style, artist, genre; the original train/test split is pooled and not used |

Terms of use: the WikiArt images may be used for non-commercial research only and are subject to the
terms of WikiArt.org; MultitaskPainting100k is subject to the terms of its authors.

Adapters: `src/stylebench/data/{wikiart,multitask_painting}.py`, both returning the canonical columns
`image_id, path, style, artist, genre`. A missing file raises `DatasetUnavailable` naming what is needed.
The label mapping and every cleaning decision are documented in `docs/PROTOCOL.md`.
