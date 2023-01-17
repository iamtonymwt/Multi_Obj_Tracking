# MultiObjTracking Project

Store CenterNet based MOT model and fpks

## Introduction
This project includes the Muti-object tracking model based on CenterNet. This repo contains traning codes, SDSP converted fpk file, and input&output tensor dimension info.

<center>
<figure class='half'>
<img width ='100', height = '250' src=./images/model_structure.png/> &emsp;
<img width ='300', height = '250' src=./images/static_demo.png/>;
</figure>
</center>
<center> <font size = 2> model tail and prediction effect </font> </center>
<br/>

## Project Demo

<center>
<video src="./images/video_DR_1.mp4"></video>
</center>



## MOT Folders
1. dataset includes the Caltech Pedestrain Dataset(which needs to be download), and its correspondent cleaning/modifying scripts
2. google_api includes the official google api code based on centernet using tensorflow2, contains the whole model and traning code. I made some changes and fixed some bugs. Its post-process part is modified heavily. Be Careful.
3. MultiObjTracking includes some scripts for trainig and eval, .config files for QAT training, scripts for pb and tflite conversion, model in tflites, scripts for SDSP convertor(have no permit to upload), and scripts to test tflite models.
4. documents includes my notes for this project and some working logs
5. fpk contains the final fpk results and model infos

## MOT Pipeline
### PIPELINE in English
1. Dataset Preparation, need to follow PSACAL VOC format
      * dataset and scripts are all in folder ./dataset. Follow the .md file as instruction
2. Converting dataset into TFRecord format using google api
      * (./MultiObjTracking/create_pascal_tf_record.py). changed a little. added reID infos
3. Run ./MultiObjTracking/lib_update to package google api into python package
4. Run ./MultiObjTracking/train.sh for traininig (using mot_qat.config for QAT training)
5. Run ./MultiObjTracking/eval.sh to evaluate
6. Run ./MultiObjTracking/tf2pb.sh to convert model to .pb files
7. Run ./MultiObjTracking/pb2tflite.py to convert .pb models to tflite
8. Run ./MultiObjTracking/tflite_test/tflite_test.py for pc simulation
9. Using scripts in ./MultiObjTracking/SDSP/ for SDSP conversion


### PIPELINE in Chinese
1. 数据集准备，需要满足规定格式（PSACAL VOC）
      * 在./dataset里有caltech数据集，和相应的转换代码，按照README运行即可
2. 调用google api的脚本完成到tfrecord的转化
      * (./MultiObjTracking/create_pascal_tf_record.py）有修改，加入了reID
3. 根据需求修改google api
      * 修改后的api已经上传gitlab
4. 运行./MultiObjTracking/lib_update完成打包
5. 运行./MultiObjTrackingtrain.sh训练
6. 运行./MultiObjTrackingeval.sh测试
7. 运行./MultiObjTrackingtf2pb转换模型成.pb
8. 运行./MultiObjTrackingpb2tflite完成quant和tflite转换
9. 运行./MultiObjTrackingtflite_test/tflite_test.py进行pc simulation
10. 运行SDSP相关脚本