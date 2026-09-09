# data/

Datasets are **not** committed. Everything under `data/` except this file and
`.gitkeep` is git-ignored.

## Middlebury 2014

Each scene directory must contain:

```
<Scene>/
  im0.png      left  image
  im1.png      right image
  calib.txt    cam0, cam1, doffs, baseline (mm), width, height
  disp0.pfm    left-view ground-truth disparity (optional but recommended)
```

Fetch one scene with the helper:

```bash
scripts/download_middlebury.sh Motorcycle
# -> data/middlebury/Motorcycle/{im0.png,im1.png,calib.txt,disp0.pfm}
```

Then:

```bash
./build/apps/benchmark_depth --source data/middlebury/Motorcycle
```

Full dataset and terms: <https://vision.middlebury.edu/stereo/data/scenes2014/>

`calib.txt` example:

```
cam0=[3997.684 0 1176.728; 0 3997.684 1011.728; 0 0 1]
cam1=[3997.684 0 1307.839; 0 3997.684 1011.728; 0 0 1]
doffs=131.111
baseline=193.001
width=2964
height=2000
ndisp=280
```

`MiddleburySource` converts `baseline` to metres, uses per-view `cx` to set
`doffs = cx1 - cx0`, and rescales `fx, fy, cx, cy, doffs` if `im0.png` was
downsized from the calibrated resolution.

## KITTI (Phase 3)

The tracking loader and MOTA/IDF1 evaluation arrive with Phase 3. Expected
layout will be documented here then.
