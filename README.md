# brainweb tissue segmentation

## Dataset

Creation dataset by [Brainweb](https://brainweb.bic.mni.mcgill.ca/) Simulator.

The Following figure shows the configs in Simulator

![alt text](https://github.com/smohammadi96/brainweb_tissue_segmentation_unet/blob/main/sample/config.PNG)

dataset generated in three mode: ""MS mild"", ""MS moderate"", """MS severe""

## Preprocessing 
1. convert 3D T1 (.mnc) to 2D
2. normalization
3. remove extra tissues from T1
4. remove extra tissues from mask

### T1 after Preprocessing
![alt text](https://github.com/smohammadi96/brainweb_tissue_segmentation_unet/blob/main/sample/dataset_sample.PNG)

### mask after Preprocessing


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
