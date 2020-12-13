# Project-DeepLearning - Identify American Sign Language in images

![](https://user-images.githubusercontent.com/49924571/102016430-e3036900-3d58-11eb-99a3-38d7e5b288df.jpg)  
*Hello in American Sign Language*

In this project we try to identify the gestures in the American Sign Language Alphabet using a dataset of somewhat low resolution older pictures.
Data is pulled directly in the notebook and preprocessing is also made available.

We further analyse wich symbols are easier to identify and which are easier to confound and over this try different appraches to improve a base model.  

We Use a CNN based model, to identify the signals in images, with over 98% accuracy on the test images, but given the limited nature of training samples, new test samples 
with different conditions are not so easily identified.  
The final report is available including other studies on the behaviour of the model.


Image dataset reference:
Pugeault, N., and Bowden, R. (2011). Spelling It Out: Real-Time ASL Fingerspelling Recognition In Proceedings of the 1st IEEE Workshop on Consumer Depth Cameras for Computer Vision, jointly with ICCV'2011
