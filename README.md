# Deeplab Seismic Implementation

The folder containg the images. The hierarchy is as follows:

```bash
datasets
├── tfrecords
├── tfrecords_test
├── Customdataset
│   ├── Imageset
|   |   ├── train.txt
|   |   ├── trainval.txt
|   |   ├── val.txt       
│   ├── JPEGImages
│   ├── SegmentationClass
├── model_checkpoints  
```
#### Download train/val Image dataset
Click [here](https://drive.google.com/open?id=1hVgifRtqOD_a-J2fCzzJG4kyu-A2jBjn) to download the Image data directly to `CustomDataset`. Make sure you have the following folder structure in the `CustomDataset` directory after you unzip the file: 

#### Download test Image dataset
Click [here](https://drive.google.com/open?id=1hVgifRtqOD_a-J2fCzzJG4kyu-A2jBjn) to download the Image data.

#### Download train/val tfrecords
Click [here](https://drive.google.com/drive/folders/1EFCPgG3Sv0emkQ6ydnOjZ9rdNItisuCS?usp=sharing) to download the Image data.

#### Download test tfrecords
Click [here](https://drive.google.com/drive/folders/1TyzFCwd6-d2jZZthG0TrqBT6h-KbatQo?usp=sharing) to download the Image data.

_________________________________________

### Generate Images from .npy file.

In deepkapha folder, there are two “.npy” files used for generating the “patch” images across cross-line. And the tensorflow-1 contains the deeplab folder inside the models-research-deeplab folder. 

Right now, only images for train.txt have been generated. You need to generate the val.txt and trainval.txt images also. 

For generating the images from val.txt and trainval.txt follow the following steps.

From the deepkapha folder, open the “dataCreation.ipynb” file. Run the last four blocks of code only. Keep a check that the path described for every folder is present and before running the code, delete all the images, if present inside the path mentioned folders. Images for train is generated. You need to generate for the trainval.txt and val.txt files. 

After the images have been generated, copy the images present inside the 

     “RotatedtrainData - trainX - labels and seismic folder” and paste it in the 
     “Tensorflow-1 - models - research - deeplab - datasets - CustomDataset” folder.  Inside the CustomDataset folder there are three folders named as “SegmentationClass” and “JPEGImages” and “ImageSets”. Paste the images of seismic and labels in the “JPEGImages” and “SegmentationClass” respectively and the “trainval.txt” and the “val.txt” in the “ImageSets folder”.

    This means that at last your “datasets” folder will be having three files - “train.txt”, “trainval.txt” , “val.txt” inside the “ImageSets” folder. The “SegmentationClass” will have the labels corresponding to the ground-truth images and the “JPEGImages” will have the original images which will be a sum of image generated from train.txt, trainval.txt and val.txt.

After this, you need to generate the tfrecord. For generating the tfrecord, navigate to the build_voc2012_data.py folder present inside the datasets folder. 
#### Generate tfrecords. 
Inside the  “build_voc2012_data.py” folder, specify the path as shown below and then run this python file. This will generate tfrecords file for the images.

tf.app.flags.DEFINE_string('image_folder', './CustomDataset/JPEGImages', 'Folder containing images.')
tf.app.flags.DEFINE_string('semantic_segmentation_folder', './CustomDataset/SegmentationClass', 'Folder containing semnatic segmentation annotations')
tf.app.flags.DEFINE_string('list_folder','./CustomDataset/ImageSets/','Folder containing lists for training and validation')
tf.app.flags.DEFINE_string('output_dir','./tfrecord','Path to save converted SSTable of TensorFlow examples.')

After the tf record, navigate to models-research-deprecated-segmentation_on_dataset.py folder
### train models 
```bash
!python train.py --logtostderr --train_split="train" --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=513,513 \
  --train_batch_size=2 \
  --training_number_of_steps=1000 \
  --fine_tune_batch_norm=true \
  --train_logdir=" path/train" \
  --dataset="seismic" \
  --dataset_dir="path/tfrecord " 
  --model_checkpoint = "path/model.ckpt" 
  ```

