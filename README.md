### csharp_nano_gpt
Spell out some nonsense c# code. A fun weekend project.

# Acknowledge
This project is a heavely modified version of the original repo by Karpathy about transformers, that you can find here: https://github.com/karpathy/ng-video-lecture. 
The core of his code is in there, I've just refactored it.

I would highly recomend watching Karpathy's video lectures, is like getting a PhD for free. You can find the syllabus at https://karpathy.ai/zero-to-hero.html  


# Goals
 The main idea behind this project is to have a client-server infrastructure to generate "nonsense" c# code given an initial prompt, "nano-copilot?".

# Data
The data come from huggingface. There is a fantastic collection of datasets published by https://huggingface.co/loubnabnl. I am specifically using this one https://huggingface.co/datasets/loubnabnl/bigcode_csharp/tree/main

The script data.py reads this dataset and create a new file with only the c# code. It remove the metadata and also remove empty lines and extra spaces from the the code. The new file is the one used for training.

# Project Structure
The main transformers network is in the file network.py. Hyperparameters are in the Config.py file as well as other configuration settings.
You could potentially run multiple experiments and compare results with different hyperparameters. The "experiement" parameter will be used to save models in the appropriate folder structure. The training script will automatically save the best and last models in the "experiments/<experiment>" folder structure and the client.py and predict.py will read the model from there as well. The "use_best_model" parameter controls whether the best or the last model will be used. This parameter affects training and inference.
The server.py file is a websockets server that will load the trained model and wait for requests. The client.py file is the counterpart to it.
Predict.py is just that, a small script to load the model and generate some code based on the provided prompt.

# Encoder
In the original lecture Karpathy'd implemented a simple encoder/decoder but suggested that we try out sentencepiece, tiktoken or any other. This projects use tiktoken from OpenAI.
