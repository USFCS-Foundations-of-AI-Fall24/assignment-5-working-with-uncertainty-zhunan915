

import random
import argparse
import codecs
import os
import numpy as np

# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""
        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        with open(f"{basename}.trans", "r") as f:
            for line in f:
                elem = line.strip().split()
                state = elem[0]
                if state not in self.transitions:
                    self.transitions[state] = {}
                for i in range(1, len(elem), 2):
                    next_state = elem[i]
                    prob = float(elem[i + 1])
                    self.transitions[state][next_state] = prob

        # Load emissions
        with open(f"{basename}.emit", "r") as f:
            for line in f:
                elem = line.strip().split()
                state = elem[0]
                if state not in self.emissions:
                    self.emissions[state] = {}
                for i in range(1, len(elem), 2):
                    next_state = elem[i]
                    prob = float(elem[i + 1])
                    self.emissions[state][next_state] = prob


   ## you do this.
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        states = []
        emissions = []
        current_state = "#"
        for step in range(n):
            if current_state in self.transitions:
                next_states = list(self.transitions[current_state].keys())
                next_probs = list(self.transitions[current_state].values())
                current_state = np.random.choice(next_states, p=next_probs)
            else:
                break
            states.append(current_state)
            if current_state in self.emissions:
                possible_emissions = list(self.emissions[current_state].keys())
                emission_probs = list(self.emissions[current_state].values())
                emission = np.random.choice(possible_emissions, p=emission_probs)
                emissions.append(emission)
            else:
                break
        return Sequence(states, emissions)

    def forward(self, sequence):
        pass
    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.






    def viterbi(self, sequence):
        pass
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hidden Markov Model Sequence Generator")
    parser.add_argument("domain", help="Domain name (e.g., cat, partofspeech, lander)")
    parser.add_argument("--generate", type=int, help="Number of states to generate")
    args = parser.parse_args()

    hmm = HMM()
    hmm.load(args.domain)

    if args.generate:
        sequence = hmm.generate(args.generate)
        print("Generated States:\n", ' '.join(sequence.stateseq))
        print("Generated Emissions:\n", ' '.join(sequence.outputseq))

    '''
        run python HMM.py cat --generate 20
        results:
            Generated States:
                grumpy grumpy happy hungry grumpy happy happy hungry grumpy grumpy happy hungry grumpy happy happy hungry grumpy hungry grumpy happy
            Generated Emissions:
                meow silent silent silent silent meow purr meow meow meow meow meow silent purr meow purr meow meow purr meow
    '''



