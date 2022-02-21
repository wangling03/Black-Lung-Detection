# Black Lung Detection project
Implemented a Deep CNN model for Detection of pneumoconioses Patients Using Chest X-Ray Images. Built the model using Transfer Learning by fine-tuning pre-trained model  DenseNet121 that have been pre-trained on ImageNet dataset.

Fifty-five Chest X-Ray diagnosed with pneumoconioses from B-reader Syllabus and sixty-one normal Chest X-Ray from CheXPert were used in the project. Ninty-two Chest X-rays (46 with pneumoconioses and 46 normal Chest X-rays) were used train the model and 24 Chest X-rays (12 with pneumoconioses and 12 normal Chest X-rays) were used as test set.

The Chest X-rays were enhanced using Bi-Histogram Equalization with Adaptive Sigmoid Functions algorithm (BEASF) and Contrast Limiting Adaptive Histogram Equalization (CLAHE). 

The confusion matrix of the test set is:

![alt text](https://github.com/wangling03/Black-Lung-Detection/blob/main/Figure%202022-02-21%20112114.png?raw=true)

The history of loss and accuracy plot for 50 epoch is:

![alt text](https://github.com/wangling03/Black-Lung-Detection/blob/main/Figure%202022-02-21%20130843.png?raw=true)


# Future works



