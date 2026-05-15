# Experiment 6: CNN Embedding w/ Cross Attention on Moves 


## Hypothesis
In the prior experiment, one note was that there was potential loss in using a CNN on the board state converting the board state to a 768 size embedding. This is due to the fact that the dimensionality of the board state is far larger than a $$d_model = 768$$ size embedding. 

To combat this, I am now converting the board state to 64 vectors of dimension $$d_model = 768$$. These embedding vectors are then cross attended with the embedding vectors per each moves. What this does is ensure that the board state is properly giving attention to each move in a way that preserves all the board information, this will hopefully improve the richness of the hidden state per each move.

I hypothesize that I will see better performance out of the model on the test sets as a result of this improvement in addition to better puzzle top-1 and top-5 accuracy. This improvement in my opinion will mainly improve puzzle performance which suffered as a result of not being able to properly process board information.


