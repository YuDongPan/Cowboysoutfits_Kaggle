# Cowboysoutfits_Kaggle
This is source code for the Kaggle Competition-Cowboyoutfits

running environment:

1、linux operating system

2、PyTorch 1.9.0

3、CUDA 10.2

4、Python 3.7.0

Operation Steps:

1、install tools package pycocotools

2、download the Kaggle Competition dataset

3、reorganize the dataset,make it consistent with yolov5 dataset standards 

   a)run transformdDatatoXml.py
   
   b)run transformDatatoYolo.py
   
   tips:make some changes if necessary
   
4、modify the configuration files cow_data.yaml(under folder data)

5、trainning:

   command:
   
   a)python train.py --data data/cow_data.yaml --cfg models/cow_yolov5s.yaml --weights weights/yolov5s.pt --batch-size 4 --epochs 100(under command line)
   
   b)%run train.py --data data/cow_data.yaml --cfg models/cow_yolov5s.yaml --weights weights/yolov5s.pt --batch-size 4 --epochs 100(under jupyter notebook)
   
6、detecting/predicting:

   a)create a folder named cow_answer
   
   a)run cow_detect_kaggle.py
   
   b)run tansformToJson.py
   
   tips:make some changes if necessary,results saved under folder cow_answer
   
7、make your prediction on colab

