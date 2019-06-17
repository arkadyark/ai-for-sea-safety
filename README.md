## Grab AI for S.E.A. - Safety Challenge
### Arkady Arkhangorodsky

Here is my attempt for the Grab AI safety challenge, which required me to create a model to predict whether a trip was dangerous or not dangerous, based on the telematics data from the rider's phone. 

 
I attempted to train an RNN, in a few different ways. First, using a simple LSTM that would output the ride classification, then using a hierarchical attention network. Neither approach was able to learn better than guessing, however, probably due to not enough data being available. So I then tried to craft some features, including maximum speed and maximum acceleration, globally and for a contiguous segment of a ride. Using these features I tried a linear regression classifier, and a small neural network, with slightly better results (~76% accuracy), but still hardly better than guessing based on the skewed data distribution.


My submission is thus left mostly incomplete, but I still wanted to share my code.
