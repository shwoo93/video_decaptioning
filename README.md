Chalearn Inpainting challenge Track 2

1st place on final test phase (challenge ID 'Sanghyun Woo')
https://competitions.codalab.org/competitions/18421#results

Final model overview

![architecture](https://user-images.githubusercontent.com/13035722/43037442-f93bbdbc-8d47-11e8-9094-35a4eb042c3a.PNG)



Preparation:
1. Install Python 3.6.4 version and Pytorch 0.3.1.post3
2. Install Dependencies
  - visdom (training loss curve visualization)
  - ffmpeg (video to png)
3. Install pretrained weight
  - https://drive.google.com/open?id=1qsR2CXHet5h9nUl67nSpf2GW_ZxytWVK

Brief code instruction:
1. Extract png files for each mp4 videos (use video_png.py)
2. Set root path (modify --root_path flag in scripts)
3. Run the code using scripts
  - scripts/train.sh (for training the final model)
    - we trained for 200 epochs (about 3days using 2 gpus, GTX 1080 ti)
  - scripts/test.sh (for testing the final model)
    - 1~2 sec per video
* Note that we attached pretrained weight of the final model at google drive.(final_model.pth)
  Please properly modify the path of pretrained weight in test.sh file for testing.
