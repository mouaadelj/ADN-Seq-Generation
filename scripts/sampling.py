import random
import math
import pandas as pd
import torch
from data_processing import encode
from train_ensemble import get_ensemble_predections
from transformers import pipeline

def rejection_sampling(start_seq, num_samples, Model, tokenizer,ensemble_models, num_bootstraps):
    samples = []
    fitness_scores = []

    while len(samples) < num_samples:
        new_seqs = generator(start_seq, num_samples, Model, tokenizer)
        for new_seq in new_seqs:
            fit = fitness_func(start_seq, new_seq, ensemble_models, num_bootstraps)
            p = min(1.0, math.exp(-fit))
            if random.uniform(0, 1) < p:
                samples.append(start_seq + new_seq)
                fitness_scores.append(fit)
    RS = pd.DataFrame({'bindings': samples, 'fitness': fitness_scores})
    RS['fitness'] = RS['fitness'].apply(lambda x: x[0][0].item())
    return RS.sort_values(by='fitness', ascending=False)

def MH_sampling(start_seq, num_samples, Model, tokenizer,ensemble_models, num_bootstraps):
    samples = []
    fitness_scores = []

    # Initialize the first sequence
    current_seq = start_seq
    current_fitness = float(fitness_func(start_seq, current_seq, ensemble_models, num_bootstraps))

    while len(samples) < num_samples:
        # Generate candidate sequences
        new_seqs = generator(current_seq, num_samples, Model, tokenizer)

        for new_seq in new_seqs:
            new_fitness = float(fitness_func(start_seq, new_seq, ensemble_models, num_bootstraps))

            # Metropolis-Hastings acceptance criterion
            acceptance_prob = min(1, new_fitness / current_fitness)
            if random.uniform(0, 1) <= acceptance_prob:
                current_seq = new_seq
                current_fitness = new_fitness

        samples.append(start_seq + current_seq)
        fitness_scores.append(current_fitness)

    # Store results in a DataFrame
    samples_df = pd.DataFrame({'full-seq': samples, 'fitness': fitness_scores})

    return samples_df.sort_values(by='fitness', ascending=False)


# Define the energy function
def fitness_func(start_seq,binding, models,num_bootstraps):
    seq= start_seq + binding
    return get_ensemble_predections(num_bootstraps, models, torch.tensor(encode(seq).values).float())

# Define the generator
def generator(start_seq, generation_nbr, Model, Tokenizer):
    Generated_bindings=[]
    classifier = pipeline("text-generation", model = Model, tokenizer =Tokenizer)
    result=classifier(start_seq, do_sample = True, num_return_sequences=generation_nbr, return_full_text=True, max_length=10, min_length=10)
    for i in range(generation_nbr-1):
       Generated_bindings.append(result[i]['generated_text'][-8:])
    return Generated_bindings


if __name__ == "__main__":

    pass