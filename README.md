# AOF-Face-Recognition

I would like to launch a piece of research on facial recognition that especially serves the AOF community. I will learn about the underlying mechanism of facial recognition (and machine learning algorithms that drives it) and the tools that are used to implement it. I will then try to train a model with pictures of members of the AOF community and write codes to recognize people in media including photos and videos. This model and tools I produce would be helpful for future projects like, for example, even more convenient check-in for school events by scanning faces. In addition, my code and model (but not any actual data from images of the school community) will be open-sourced on GitHub. Other people and their projects can benefit from what I have created. It would encourage our students to use what they have learned in class to do great contributions to the community.

# How to use it

__All script should be run at the root directory of this project.__

Run script under the demo folder. Now the system can recognized students enrolled during 2015-2019.

###To train new models:
1. Remove all files under the data folder but keep the folders themselves.
2. Put photos under training_data/known. Change the function transform_yearbook_photos in util/general.py to convert filename into actual name displayed.
3. Change settings on top in processor.py and processor_num_map.py if needed. Run processor_num_map.py to generate encoding data for training examples. These information will be stored to data/cache/preprocess
4. Use add function in test_data/test_manager.py to add at least one test set. Change parameters on top if needed.
5. Choose a model to use. In the folder, run xxx_test.py to generate the model and test it. Model files will be generated in data/model/xxx
6. Create a EncodingsPredictor of that model. Set the correct parameters. Now you can use it with demo or with your own code.

###Testing:
There are tests for each model as well as time_test.py and all_test.py to test multiple models.

###Model Suggestions:
Usage
- Universal: NN
- Low Res: NN
- Prevent False Positive: NN
- Prevent False Negative: KNN
- Tens of thousands of training data: NN or KNNKmeans

Training
- Mode recognition speed: NN < KNNKmeans < KNN
- Mode recognition speed (huge amount of training data): NN < KNNKmeans << KNN
- Model accuracy: NN = KNN > KNNKmeans
- Model accuracy (huge amount of training data): NN = KNN = KNNKmeans
- Model training speed and hardware requirement: KNN < KNNKmeans << NN


How well do the current model do:
![Demo](https://raw.githubusercontent.com/DEDZTBH/AOF-Face-Recognition/master/pics/demo.gif)
![Accuracy](https://raw.githubusercontent.com/DEDZTBH/AOF-Face-Recognition/master/pics/accuracy.png)
![Time](https://raw.githubusercontent.com/DEDZTBH/AOF-Face-Recognition/master/pics/time.png)
![False Negative](https://raw.githubusercontent.com/DEDZTBH/AOF-Face-Recognition/master/pics/fn.png)
![False Positive](https://raw.githubusercontent.com/DEDZTBH/AOF-Face-Recognition/master/pics/fp.png)