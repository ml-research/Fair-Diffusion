import torch
import os
import json
import numpy as np
import random
from utils import get_random, face_existing
import dlib
import re
import argparse
from semdiffusers import SemanticEditPipeline

parser = argparse.ArgumentParser(description='generate images')
parser.add_argument('--mode', default='generate', type=str, choices=['generate','edit'],
                    help='which edit to conduct')
parser.add_argument('--split', default=0, type=int,
                    help='split occupations into chunks to make parallel computation possible')
args = parser.parse_args()

    
def chunks(xs, n):
    n = max(1, n)
    return list(xs[i:i+n] for i in range(0, len(xs), n))


cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
device = 'cuda'

pipe = SemanticEditPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
).to(device)

gen = torch.Generator(device=device)
num_im = 250
chunk_size = 15

prompt = 'A photo of the face of a '

with open('occupations.txt') as f:
    occupations = [line.rstrip() for line in f]

occupations = chunks(occupations, chunk_size)[args.split]

if args.mode == 'generate':
    for cl in occupations:
        pth = f"generated_images/gender_imgs_{model_name[-3:]}/{cl}"
        os.makedirs(pth, exist_ok=True)
        i, j = 0, 0
        while j < num_im:
            gen.manual_seed(i)
            params = {'guidance_scale': 7,
                      'seed': i,
                      'prompt': prompt + cl,
                      'num_images_per_prompt': 1
                     }
            out = pipe(**params, generator=gen)
            image = out.images[0]
            # check if face exists in img with fairface detector
            if face_existing(np.array(image), cnn_face_detector)==1:
                image.save(f"{pth}/image{j}.png")
                with open(f"{pth}/image{j}.json", 'w') as fp:
                    json.dump(params, fp)
                j += 1
            else:
                print(f'no Face - {i}')
            i += 1
            
            
elif args.mode == 'edit':
    dir_ = [True, False]      
    edit1 = ['male person', 'female person']
    edit2 = edit1[::-1]

    for cl in occupations:
        sampler = get_random(num_im)
        pth_edit = f"generated_images/gender_edited_imgs_{model_name[-3:]}/{cl}"
        os.makedirs(pth_edit, exist_ok=True)
        pth = f"generated_images/gender_imgs_{model_name[-3:]}/{cl}"
        for i in range(0, num_im):
            # in which direction to edit
            if sampler[i]:
                edit = edit1
            else:
                edit = edit2
            # load same params from the previously generated image that is edited now
            with open(f'{pth}/image{i}.json', 'r') as f:
                params = json.load(f)
            gen.manual_seed(params['seed'])
            params_edit = {'guidance_scale': params['guidance_scale'],
                      'seed': params['seed'],
                      'prompt': params['prompt'],
                      'num_images_per_prompt': params['num_images_per_prompt'],
                      'editing_prompt': edit,
                      'reverse_editing_direction': dir_,
                      'edit_warmup_steps': 5,
                      'edit_guidance_scale': 4,
                      'edit_threshold': 0.95, 
                      'edit_momentum_scale': 0.5,
                      'edit_mom_beta': 0.6}
            out = pipe(**params_edit, generator=gen)
            image = out.images[0]
            image.save(f"{pth_edit}/image{i}.png")
            with open(f"{pth_edit}/image{i}.json", 'w') as fp:
                json.dump(params_edit, fp)
                
