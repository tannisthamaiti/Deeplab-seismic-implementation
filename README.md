# Deeplab Seismic Implementation

There are two folders named as - Deepkapha and tensorflow-1

In deepkapha folder, there are two “.npy” files used for generating the “patch” images across cross-line. And the tensorflow-1 contains the deeplab folder inside the models-research-deeplab folder. 

Right now, only images for train.txt have been generated. You need to generate the val.txt and trainval.txt images also. 

For generating the images from val.txt and trainval.txt follow the following steps.

From the deepkapha folder, open the “dataCreation.ipynb” file. Run the last four blocks of code only. Keep a check that the path described for every folder is present and before running the code, delete all the images, if present inside the path mentioned folders. Images for train is generated. You need to generate for the trainval.txt and val.txt files. 

After the images have been generated, copy the images present inside the 

     “RotatedtrainData - trainX - labels and seismic folder” and paste it in the 
     “Tensorflow-1 - models - research - deeplab - datasets - CustomDataset” folder.  Inside the CustomDataset folder there are three folders named as “SegmentationClass” and “JPEGImages” and “ImageSets”. Paste the images of seismic and labels in the “JPEGImages” and “SegmentationClass” respectively and the “trainval.txt” and the “val.txt” in the “ImageSets folder”.

    This means that at last your “datasets” folder will be having three files - “train.txt”, “trainval.txt” , “val.txt” inside the “ImageSets” folder. The “SegmentationClass” will have the labels corresponding to the ground-truth images and the “JPEGImages” will have the original images which will be a sum of image generated from train.txt, trainval.txt and val.txt.

After this, you need to generate the tfrecord. For generating the tfrecord, navigate to the build_voc2012_data.py folder present inside the datasets folder. 

Inside the  “build_voc2012_data.py” folder, specify the path as shown below and then run this python file. This will generate tfrecords file for the images.

tf.app.flags.DEFINE_string('image_folder', './CustomDataset/JPEGImages', 'Folder containing images.')
tf.app.flags.DEFINE_string('semantic_segmentation_folder', './CustomDataset/SegmentationClass', 'Folder containing semnatic segmentation annotations')
tf.app.flags.DEFINE_string('list_folder','./CustomDataset/ImageSets/','Folder containing lists for training and validation')
tf.app.flags.DEFINE_string('output_dir','./tfrecord','Path to save converted SSTable of TensorFlow examples.')

After the tf record, navigate to models-research-deprecated-segmentation_on_dataset.py folder and continue following the analytics vidhya blog. 

