# PPG
This is a method to find the heart rate (HR) of a subject via signal processing techniques

The program will track the face of the subject and determine his forehead region.

The program will run by default for 5 times the buffer duration unless the user changes the value of "NB_OF_SAMPLES" on line 19 in the file "main.cpp". 

At the end of this duration, it will automatically display the average HR over the whole test duration for 2 different ROI: the face and the forehead of the subject. If the user wants to see their instantaneous HR, they can uncomment the lines 380 and 381 of the file "main.cpp".
