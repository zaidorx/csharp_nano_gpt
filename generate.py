import os
import random
import sys
import torch
from config import Config
from network import BigramLanguageModel

class Generator(object):

    def __init__(self):
        self.device = Config.device
        model_name = "c_sharp.tar"
        best_model_name = "c_sharp_best.tar"
        use_best_model = Config.use_best_model
        experiment = Config.experiment

        self.enc = Config.enc

        model = BigramLanguageModel()
        self.m = model.to(self.device)
        # print the number of parameters in the model
        print(sum(p.numel() for p in self.m.parameters())/1e6, 'M parameters')

        #load model weights if exists
        model_to_load = best_model_name if use_best_model==True else model_name
        path = os.path.join('experiment', experiment)        
        model_to_load = os.path.join(path, model_to_load)
        if not os.path.exists(model_to_load):
            print(f"model : {model_to_load} nor found. Exiting")
            raise Exception(f"model : {model_to_load} nor found. Can't continue.")
        best_loss = float("inf")
        val_losess = []
        if os.path.exists(model_to_load):
            checkpoint = torch.load(model_to_load, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch_init = checkpoint['epoch']
            val_losess = checkpoint['val_losses']
            best_loss = min(val_losess)
            print(f"loaded model from {model_to_load}. Trained for {epoch_init} epochs with best loss of {best_loss}")
        else:
            print(f"no model found at {model_to_load}")
            exit

        model.eval()

    def generate(self, sentence=None, random_seed = False):
        # generate from the model
        print("Generating ....")
        if sentence is None:
            start_text = self.enc.encode(' ') #start with awhite space
        else:
            start_text = self.enc.encode(sentence)
        print(f"Prompt length {len(start_text)}")
        max_num_tokens = 200
        if (max_num_tokens - len(start_text) < 10):
            max_num_tokens = len(start_text) + 10
        print(f"Generating {max_num_tokens - len(start_text)} new tokens")
        data = torch.tensor(start_text, dtype=torch.long, device=self.device)
        data = torch.reshape(data, (1,len(data)))       
        
        

        # Set this to False to always get the same answer.
        randomize_seed = random_seed

        if randomize_seed:
            # Use a random seed to make sure we get a different answer every time.        
            seed = random.randint(0, sys.maxsize)                    
            torch.manual_seed(seed)
        else:
            # Use the same seed for consistent results
            seed = 1347 #1347, seed used during training
            torch.manual_seed(seed)
        print(f"Using seed {seed}")

        results = self.m.generate(data, max_new_tokens=max_num_tokens)
        decoded = self.enc.decode(results[0].tolist())
        print(decoded + "\n\n")
        return decoded

