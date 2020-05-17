# Arabic-OCR
Generating arabic written texts from arabic scanned images by training a model

# Project pipleline
1. Data preprocessing
  - Take image as an input
  - Segment image to lines
  - Segment line to words
  - Segment words to characters
  
 Done using some image processing techniques: Thresholding , rotating , erosion and dilation , filtering to reduce noise  and skeletonization
  

2. Feature extraction
Used
* Pixel vector : the letter is fitted into a 28x28 image and every pixel represents a feature which is appended in a vector 
* Horizontal and vertical projection
Tested but not yet used
* letter width and height
* dots count
* holes count




3. Labeling 
The segmented character is mapped to its label and its feature vector .

4. Model and training
A multilayer perceptron (MLP) is used which is a class of feedforward artificial neural network .

# Enhancements and Future work.
The segmentation accuracy needs to be improved also, using ngram model will minimize the  classification error .
