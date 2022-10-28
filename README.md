# Facial-Emotion-Recognition
FER or Facial Expression Recognition

## Models
(#): Current using
### A. Detect face with:
#### Dlib HoG Face Detection
The model is good and fast but old and the accuracy is not as good as the modern models. In addition, poor operation or errors with respect to inclination angles adversely affect the final result.

#### Mediapipe Deep Learning-based Face Detection (#)
Deep learning model provided by google, fast and modern, great accuracy but sometimes too much detail slows down the model or is unnecessary.
 
### B. Training facial emotion with:
#### Support Vector Machines (#)
Based on some previous studies, it is said to be quite accurate in terms of facial emotion recognition. Experimentally it seems so, but for some emotions with similar proportions, it works quite poorly or not recognizable.

## Navigation
### 1. Prepare the emotions dataset
- The two sets of modules used are dlib and mediapipe:
  + dlib: `modules/dlib`
  + mediapipe: `modules/mediapipe`

- Both include the file `dataset_prepare.py` which is used for extracting facial proportions.
- Designed model in `dlib_FaceLandmarks.py` or `mediapipe_FaceLandmarks.py` depending on which model.
- Finally `ratio_calc.py` is used to calculate facial proportions for different emotions. The details of the scale are written in the file.
- Both output are stored in `dataset/*.csv` file.

### 2. Training the emotions dataset
- The file `main.py` is used to capture the camera as well as call the model SVM.
- Using SVM for training and prediction is done in the file `modules/svm.py`.
- `modules/fps.py` is used to display the obtained FPS.
- Finally, the face will be displayed and the result + prediction will be printed to the Terminal.