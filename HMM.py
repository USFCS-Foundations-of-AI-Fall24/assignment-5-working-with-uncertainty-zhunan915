

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
        if "#" not in self.transitions:
            raise ValueError("Initial state '#' not defined.")

        next_states = list(self.transitions["#"].keys())
        next_probs = list(map(float, self.transitions["#"].values()))
        current_state = np.random.choice(next_states, p=next_probs)

        for elem in range(n):
            states.append(current_state)
            if current_state in self.emissions:
                possible_emissions = list(self.emissions[current_state].keys())
                emission_probs = list(map(float, self.emissions[current_state].values()))
                emission = np.random.choice(possible_emissions, p=emission_probs)
                emissions.append(emission)
            else:
                break
            if current_state in self.transitions:
                next_states = list(self.transitions[current_state].keys())
                next_probs = list(map(float, self.transitions[current_state].values()))
                current_state = np.random.choice(next_states, p=next_probs)
            else:
                break

        return Sequence(states, emissions)

    def forward(self, sequence):
        forward_prob = [{}]

        for state in self.transitions["#"]:
            forward_prob[0][state] = (
                    self.transitions["#"].get(state, 0) * self.emissions[state].get(sequence[0], 0)
            )
        for t in range(1, len(sequence)):
            forward_prob.append({})
            for current_state in self.transitions:
                if current_state in self.emissions:
                    prob_sum = sum(
                        forward_prob[t - 1][prev_state] *
                        self.transitions[prev_state].get(current_state, 0) *
                        self.emissions[current_state].get(sequence[t], 0)
                        for prev_state in forward_prob[t - 1]
                    )
                    forward_prob[t][current_state] = prob_sum
        final_probs = forward_prob[-1]
        most_prob_state = max(final_probs, key=final_probs.get)
        total_prob = sum(final_probs.values())
        norm_prob = final_probs[most_prob_state] / total_prob if total_prob > 0 else 0

        return most_prob_state, norm_prob






    def viterbi(self, sequence):
        viterbi_prob = [{}]
        path = {}

        for state in self.emissions:
            viterbi_prob[0][state] = (
                    self.transitions["#"].get(state, 0) * self.emissions[state].get(sequence[0], 0)
            )
            path[state] = [state]

        for t in range(1, len(sequence)):
            viterbi_prob.append({})
            new_path = {}

            for current_state in self.emissions:
                max_prob, best_prev_state = max(
                    (viterbi_prob[t - 1][prev_state] *
                     self.transitions[prev_state].get(current_state, 0) *
                     self.emissions[current_state].get(sequence[t], 0), prev_state)
                    for prev_state in self.emissions
                )
                viterbi_prob[t][current_state] = max_prob
                new_path[current_state] = path[best_prev_state] + [current_state]

            path = new_path

        max_final_prob, best_final_state = max((viterbi_prob[-1][state], state) for state in self.emissions)
        return path[best_final_state]





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hidden Markov Model Sequence Generator")
    parser.add_argument("domain", help="Domain name (cat, partofspeech, lander)")
    parser.add_argument("--generate", type=int, help="Number of states to generate")
    parser.add_argument("--forward", help="File with sequence of emissions to run forward algorithm")
    parser.add_argument("--output_file", help="Output file to save generated emissions", default="generated_sequence.obs")
    parser.add_argument("--viterbi", help="File with sequence of emissions to run viterbi algorithm")
    args = parser.parse_args()

    hmm = HMM()
    hmm.load(args.domain)

    if args.generate:
        sequence = hmm.generate(args.generate)
        print("Generated States:\n", ' '.join(sequence.stateseq))
        print("Generated Emissions:\n", ' '.join(sequence.outputseq))

        with open(args.output_file, "w") as f:
            line_length = 5
            i = 0
            while i < len(sequence.outputseq):
                emission_chunk = sequence.outputseq[i:i + line_length]
                f.write(' '.join(emission_chunk) + " .\n")
                i += line_length
                line_length = random.randint(3, 6)

            print(f"Generated emissions saved to {args.output_file}")

    '''
        run python hmm.py partofspeech --generate 20 --output_file speech_parts.obs
        run python HMM.py cat --generate 20 
        results:
            Generated States:
                grumpy grumpy happy hungry grumpy happy happy hungry grumpy grumpy happy hungry grumpy happy happy hungry grumpy hungry grumpy happy
            Generated Emissions:
                meow silent silent silent silent meow purr meow meow meow meow meow silent purr meow purr meow meow purr meow
        run python hmm.py cat --generate 20 --output_file cat_sequence.obs
        run python hmm.py lander --generate 20 --output_file lander_sequence.obs
    '''

    if args.forward:
        safe_spots = ["4,3", "3,4", "4,4", "2,5", "5,5"]
        with open(args.forward, "r") as f:
            for line in f:
                emissions = line.strip().split()
                if emissions:
                    final_state, probability = hmm.forward(emissions)
                    print(f"SEQUENCE: {emissions}")
                    print(f"FINAL PREDICTED STATE: {final_state}")
                    print(f"PROBABILITY: {probability}")
                    if args.domain == "lander":
                        status = "SAFE" if final_state in safe_spots else "NOT safe"
                        print(f"Lander is: {status}")

    if args.viterbi:
        with open(args.viterbi, "r") as f:
            for line in f:
                emissions = line.strip().split()
                if emissions:
                    most_likely_sequence = hmm.viterbi(emissions)
                    print(f"{' '.join(most_likely_sequence)}")
                    print(f"{' '.join(emissions)}")

    '''
        run -> python hmm.py cat --viterbi cat_sequence.obs    
            SEQUENCE: silent silent meow meow silent .
            LIKELY STATE SEQUENCE: grumpy grumpy happy hungry hungry 
            
            SEQUENCE: purr meow purr .
            LIKELY STATE SEQUENCE: happy hungry hungry 
            
            SEQUENCE: purr purr meow .
            LIKELY STATE SEQUENCE: happy happy hungry 
            
            SEQUENCE: purr silent silent purr .
            LIKELY STATE SEQUENCE: happy happy happy hungry 
            
            SEQUENCE: purr purr silent meow silent .
            LIKELY STATE SEQUENCE: happy happy happy hungry hungry   
        
        run -> python hmm.py partofspeech --viterbi ambiguous_sents.obs
            PRON VERB DET NOUN .
            i shot the elephant .
            PRON VERB DET NOUN ADP DET NOUN .
            he took my shot at the elephant .
            NOUN VERB ADP DET NOUN .
            flies waited at the window .
            DET NOUN VERB DET NOUN .
            the pilot flies the plane .
            DET VERB DET ADJ NOUN .
            this is a light blanket .
            PRON VERB DET NOUN PRT .
            she turned the light off .
            DET NOUN NOUN DET NOUN .
            the lanterns light our path .
            VERB PRON VERB PRON .
            did you train her ?
            DET NOUN VERB VERB ADV .
            the train is arriving now .
            PRON NOUN DET NOUN .
            they book the ticket .
            PRON VERB DET NOUN .
            i love this book !
    '''
