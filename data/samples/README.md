# Sample clips

Video files in this directory are git-ignored. Only use footage you have the right to use.

The clip used during development can be fetched manually from Wikimedia Commons:

- **File:** [FC Utrecht takes a Corner.ogv](https://commons.wikimedia.org/wiki/File:FC_Utrecht_takes_a_Corner.ogv)
- **Author:** Pel Laurens
- **License:** [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)

```bash
curl -L -o data/samples/fc_utrecht_corner.ogv \
  https://upload.wikimedia.org/wikipedia/commons/7/7a/FC_Utrecht_takes_a_Corner.ogv
```

Convert it to MP4 (H.264, video only) with the bundled PyAV FFmpeg. Run this from the repository root with the project installed in the active environment, because the script imports `soccervision`:

```bash
python scripts/convert_to_mp4.py "data/samples/fc_utrecht_corner.ogv" data/samples/fc_utrecht_corner.mp4
```
