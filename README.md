# Animal-Classification-Using-Deep-Learning USING Pytorch 

Animals-10 Dataset
As a image dataset have been used Animals-10 dataset available at kaggle (https://www.kaggle.com/alessiocorrado99/animals10 ). 
It contains about 28K medium quality animal images grouped into 10 classes

Data Processing
In order to prepare data for learning images should be divided to 3 subsets: train dataset, validation dataset and test dataset.
Firstly, one dataset had been created with function ImageFolder from datasets module. Resize((64,64)) and ToTensor() functions were used as transform.
Then custom dataset was created, with init, len and getitem functions.

Next step was splitting original dataset into 3 subsets, and then making final sets (training, validation and test) with chosen transforms. 
In the end, validation and test set contained about 2,600 images, where train set had about 20,000 images.
Finally created 3 dataloaders with batch size of 128.

Defining the Model
Defining BaseModel class give option to try different neural networks architectures, while keeping the same basic functions to training and evaluate the model. 
In this case, all tested models were extensions of BaseModel.

Best results were achieved when using ResNet9 as classification model. Architecture of residual network is shown inte Notebook file

Training the Model
Three functions defined below were responsible for training process: adjusting weights of the model, 
evaluate performance based on validation set, and getting value of learning rate. As can be seen, used optimizer was Adam. 
To provide better training, gradient clipping and adaptive learning rate were applied in training phase.

Model was trained on GPU in order to speed up learning process. Few approaches were taken, and finally undermentioned hyperparameters were chosen for learning

Results
Model was trained for 15 epochs, and for each cross entropy loss for training and validation set were calculated. 
Accuracy for validation set were calculated and saved as well. After training loss functions and accuracy for each epoch were plotted.
As can be seen on first plot, both training loss and validation loss decreased during learning. Validation loss had some spikes on the way, but at last epoch, 
has lower value than training loss. From that, it is certain, that model doesn’t overfit the data.
Moreover, achieved accuracy come to 0.86 (86%), what is really nice result.
Presented results are the best from tested approaches. Other architectures, training without data augmentation, testing different values of maximum learning rate yielded to worse performance of classification. Presented approach isn’t probably the best possible, but results comes quite close to the state of the art performance.
