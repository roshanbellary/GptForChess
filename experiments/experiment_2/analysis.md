## Experiment 2: My Model Is Low IQ

Played against the reward model with a couple of different positions. Start game has vastly improved from the stockfish training which is great :) However...model has horrible endgame and misevaluates its position entirely when it comes to endgame. I had model
in a checkmate position in 3 moves; however, model continually thought it had +0.9 odds of winning. Thus, I need to reevaluate how
I'm doing training for the model.

## Takeaways/Next Steps

First, I need to surface games that feature endgames to add to model repetoire placing weight on those games lets it learn endgame better. Second, I need a larger diversity of Stockfish games to train on(eventually I might end up cutting out the training on +1/0/-1) loss entirely due to its highly noisy resolution. Training on such a large set with a crazy amount of time spent on epochs is rather stupid and costly. Going to cut down on that dataset size and bulk up Stockfish size. Finally, since I anyways need to redo
phase 1 training, I'm going to fix up tokenizer to take in any possible chess move rather than lie rely upon corpus(I'll just append the missing moves to corpus).