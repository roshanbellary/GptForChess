## Experiment 1: Initial Training and Notes


Trained model for 10 epochs on 20M games for 17 hours on a gpu from vast. Phase 1 training took 15 hours, Phase 2 took around 2 hours. Noted the following about the loss

Very very noisy loss during phase 1 as epoch loss drifted down close to 0.6. This arises from the noisy labels applied to game results.
High Bayes Error -> Noisy Data and Noisy Loss


During phase 2 training on Stockfish centipawn scores normalized to fit b/w -1 and 1, model had steadily decreasing task loss(it was improving at guessing Stockfish)
however distill loss steadily crept higher. This makes sense since model was slowly leanring to evaluate start game which occurs w/i first 10 moves. However, after playing
model with demo.py I still noted that reward score was very very inaccurate during start game due to model not sufficiently learning how to evaluate positions in beginning.

In addition, I made a dumb mistake while building tokenizer on 20M chess games as I made the assumption that surely all possible moves that would feasibly be committed
in game are covered by this dataset. This turned out to be only partially true as out of the hundreds of games that I played with the minimax wrapper on top of reward model. One game had a move that didn't lie in corpus thus causing model to crash. Thus, need to run another phase 1 training to cover these rarer moves(My poor wallet 😭). 


For second experiment, I'm decreasing distill factor by 80% to lambda = 0.01 and jacking up epochs on phase 2(this is still very very cheap to train on gpus). I'll do another
phase 1 training later on after fixing phase 2 to be smarter and better.