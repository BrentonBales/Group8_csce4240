CSCE 4240 Project

Group 8 is:
    Nolan Wilkins
    John Greig
    Brenton Bales

Requirements:
    - Have a webcam plugged in when the project is run
    - Python3 v. 3.12.1 or better
    - Python libraries:
        - opencv-contrib-python, numpy, sys, os
        - sys and os are default, so no installation is needed for those two
        - you might need to uninstall opencv and replace it with opencv-contrib-python if opencv is already installed

Setup:
    1. Enroll the people you want to recognize:
    $   python face_enroll.py
        - Enter person's name when prompted
        - Look at camera while it captures 30 face samples
        - Repeat this process for each person you want to recognize

    2. If you skip training during enrollment, train manually:
    $   python trainModel.py
        - Make sure the file structure is like:
            - database
                - raw_faces
                    - personName
                        - <images of person>
                    - personName
                        - <images of person>

Run:
    - Open your terminal to the directory with main.py
    - Depending on your installation, running may be in the ballpark of:
    $   py ./main.py
    $   python3 ./main.py
    - Green boxes with people = recognized people
    - Red boxes with "Unknown" = unrecognized people
    - Number in parentheses show confidence
    - Press Q to quit

Output:
    - Detected faces are saved to
        - detectedFaces
            - personName
                - instanceNumber
                    - <images of person>
            - personName
                - instanceNumber
                    - <images of person>

Adding more people:
    1. Run enrollment again:
    $   python face_enroll.py

    2. Retrain the model to include new person:
    $   python trainModel.py