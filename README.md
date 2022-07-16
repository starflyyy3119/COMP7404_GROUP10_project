---

Original Project: https://github.com/facebookresearch/VideoPose3D

---


# Dependencies

- Python 3+ distribution
- [Pytorch](https://pytorch.org/get-started/previous-versions/) v1.10.0 (for compatibility with Detectron2)
- Cuda v11.3 (for compatibility with Detectron2)

# Evaluating the pretrainied model

- Download the preprocessed Human3.6 DataSet npz files(download [data_2d_h36m_cpn_ft_h36m_dbb.npz](https://drive.google.com/file/d/1FfnFpFzoOsJ2kzaY_L9q8buS7t2GClBL/view?usp=sharing) and [data_3d_h36m.npz](https://drive.google.com/file/d/1mFAlUkeIUCguSvYLib21l6VY6ByVFsQk/view?usp=sharing) put it in the `data` folder)
- Download [pretrained_h36m_cpn.bin] and put it into the `checkpoint` folder.
- You can then test the model on Human 3.6m DataSet.

```
gdown --fuzzy https://drive.google.com/file/d/1FfnFpFzoOsJ2kzaY_L9q8buS7t2GClBL/view?usp=sharing -O data/data_2d_h36m_cpn_ft_h36m_dbb.npz
gdown --fuzzy https://drive.google.com/file/d/1mFAlUkeIUCguSvYLib21l6VY6ByVFsQk/view?usp=sharing -O data/data_3d_h36m.npz
wget -c https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_cpn.bin -O checkpoint/pretrained_h36m_cpn.bin
python run.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin
```

# Run on custom videos

## 1. Set up

- Download the [pre-trainined model](https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin) and put it into the checkpoint direcotry (the file name is `pretrained_h36m_detectron_coco.bin`).

```shell
wget -c https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin -O checkpoint/pretrained_h36m_detectron_coco.bin
```

## 2. Prepare Input Video

- Prepare videos and put them into video folder.
- (Optional) Use ffmpeg library to cut the specific part of the video.
E.g.

```shell
pip install ffmpeg
ffmpeg -i video_raw/video1_raw.mp4 -ss 0:11 -to 0:50 -c copy video/video1.mp4
```

## 3. Set up Detectron2

- Set up [Detectron2](https://github.com/facebookresearch/detectron2). Prior installing Detectron2, specific version Pytorch and Cuda are installed

```shell
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge -y
```

```shell
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

## 4. Infer 2D keypoints with Detectectron

- Run infer_video_d2.py within inference folder. All files with the specified extenstion will be infered 2D keypoints.

```shell
cd inference
python infer_video_d2.py \
    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
    --output-dir ../inference_output_dir \
    --image-ext mp4 \
    ../video
cd ..
```
All files under ../video, with extension mp4, will be infered and output to ../inference_output_dir

## 5: Create a custom dataset based on the 2D keypoint inference

- Run our dataset preprocessing script from the data directory

```shell
cd data
python prepare_data_2d_custom.py -i ../inference_output_dir -o myvideos
cd ..
```
This creates a custom dataset named myvideos (which contains all the videos in inference_output_dir, each of which is mapped to a different subject) and saved to data_2d_custom_myvideos.npz. You are free to specify any name for the dataset.

Note: as mentioned, the script will take the bounding box with the highest probability for each frame. If a particular frame has no bounding boxes, it is assumed to be a missed detection and the keypoints will be interpolated from neighboring frames.

## 6: Render a custom video and exporting coordinates
- You can finally use the visualization feature to render a video of the 3D joint predictions. You must specify the custom dataset (-d custom), the input keypoints as exported in the previous step (-k myvideos, or the dataset name that you specified), the correct architecture/checkpoint, and the action custom (--viz-action custom). The subject is the file name of the input video, and the camera is always 0.

```shell
python run.py \
    -d custom \
    -k [dataset name] \
    -arc 3,3,3,3,3 \
    -c checkpoint \
    --evaluate pretrained_h36m_detectron_coco.bin \
    --render \
    --viz-subject [input video filename with extension] \
    --viz-action custom \
    --viz-video [input video file path] \
    --viz-output [output file path] \
    --viz-size 6
```

You can also export the 3D joint positions (in camera space) to a NumPy archive. To this end, replace --viz-output with --viz-export and specify the file name.

## Limitations and tips
- The model was trained on Human3.6M cameras (which are relatively undistorted), and the results may be bad if the intrinsic parameters of the cameras of your videos differ much from those of Human3.6M. This may be particularly noticeable with fisheye cameras, which present a high degree of non-linear lens distortion. If the camera parameters are known, consider preprocessing your videos to match those of Human3.6M as closely as possible.
- If you want multi-person tracking, you should implement a bounding box matching strategy. An example would be to use bipartite matching on the bounding box overlap (IoU) between subsequent frames, but there are many other approaches.
- Predictions are relative to the root joint, i.e. the global trajectory is not regressed. If you need it, you may want to use another model to regress it, such as the one we use for semi-supervision.
- Predictions are always in camera space (regardless of whether the trajectory is available). For our visualization script, we simply take a random camera from Human3.6M, which fits decently most videos where the camera viewport is parallel to the ground.
