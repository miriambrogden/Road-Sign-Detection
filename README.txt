CIS 4720 ASSIGNMENT 3
ROAD SIGN SEGMENTATION
ABHISHEK JHOREE, MIRIAM SNOW, AREEB MUSTAFA
APRIL 2019

--------------------------------------------------------

PREREQUESITES:
    - Have Python 2.7x installed on the machine
    - Have opencv version > 3.0.0 installed on the machine
    - Ensure line 159 of text_recognition.py is pointing to the correct path

RUN:
    - Open terminal in the main project folder directory
    - Type 'python algorithm.py TestImages/$NAME_OF_IMAGE'
    - If you wish to include a video to process type 'video' as the 2nd argument
    - For example, 'python algorithm.py TestImages/$NAME_OF_VIDEO video'

FOLDERS:
    - Ouput images for color segmentation and shape detection are placed in the 'ouputs' folder
    - Video frames are placed in the 'video' folder
    - Test images and videos used for experimentation are placed in the 'Test Images' folder

PURPOSE:
For the purpose of this assignment, we implemented a road sign detection algorithm to identify, locate and interpret road signs in images. We decided to extend the assignment past still images to video analysis as well. After identifying the road sign, interpretation of road signs was another key aspect of this algorithm. This included translation of non-english signs, and outputting instructions to the user based on the text recognized on the signs.
