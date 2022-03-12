# Hand-Gesture-Recognition
Ability to identify simple hand gestures through machine learning prediction

In this project, we try to identify simple hand gestures by extracting data from flex and emg sensors and predicting them through ML based algorithms. 

## Dataset:

<img width="305" alt="Screenshot 2022-03-12 151211" src="https://user-images.githubusercontent.com/66628385/158012948-db9d1fde-e9d6-49ae-9a85-a8b431e3c99b.png">
image source: google images

Using the following diagram, we record the gesture data from flex and emg sensor. Only gestures J and Z are discarded as they involve motion. The data has the following comparisons:
Total Features: 20
Class labels: 24
Delay: 0.25 secs

## Workflow:




<img width="861" alt="Screenshot 2022-03-12 150506" src="https://user-images.githubusercontent.com/66628385/158014197-36935ab5-092b-4c3d-a749-57bc9fa5bdfd.png">

The diagram here shows the workflow carried out to predict hand gestures using the sensors. Here, EMG was not used in prediction as data showed minor change which was not enough for ML based prediction. The entire recognition was done using flex sensors.

## Results

<img width="611" alt="Screenshot 2022-03-12 155326" src="https://user-images.githubusercontent.com/66628385/158014266-b0505a92-d5dc-48cb-b151-1e9e64a9d7b9.png">

The above diagram shows the performance of different algorithms. The best was noted for Adversarial Learning (AL).

