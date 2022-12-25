# brainweb tissue segmentation

## Dataset

Creation dataset by [Brainweb](https://brainweb.bic.mni.mcgill.ca/) Simulator.

## Preprocessing 

## Train

1. **Install dependencies:**


```
 pip install -r requirements.txt
```

2. **Training**

```
python train.py --dataset /path/to/hdf5 --BS_train batch_size --BS_test batch_size --img_h image_height --img_w image_width --epochs epochs_number --model_name /model/path
```

Example:**

```
python train.py --dataset sample/dataset_without_noise_csf_1.hdf5 --BS_train 32 --BS_test 32 --img_h 256 --img_w 176 --epochs 40 --model_name csf_tissue
```

## Evaluation

```
python test.py --images /path/to/images/T1  --labels /path/to/groundtruth --model_path /path/to/model --save_output True/False
```

## Results

|   | CSF | GM | WM |
|-------|-------|-------|-------|
| mIOU | 0.69	| 0.83	| 0.72 |


![alt text](https://github.com/smohammadi96/brainweb_tissue_segmentation_unet/blob/main/sample/result.PNG)
