Chalearn Inpainting challenge Track 2

1st place on final test phase (challenge ID 'Sanghyun Woo')
https://competitions.codalab.org/competitions/18421#results

Preparation:
1. Install Python 3.6.4 version and Pytorch 0.3.1.post3
2. Install Dependencies
  - visdom (training loss curve visualization)
  - ffmpeg (video to png)
3. Install pretrained weight
  - https://drive.google.com/open?id=1qsR2CXHet5h9nUl67nSpf2GW_ZxytWVK

Brief code instruction:
1. Extract png files for each mp4 videos (use video_png.py)
2. Set root path (modify opt.py)
3. You can run the code using scripts
  - scripts/train.sh (for training the final model)
  - scripts/test.sh (for testing the final model)
   
* Note that we attached pretrained weight of the final submitted model for test.(final_model.pth)
  Properly modify the path of pretrained weight in test.sh file.
