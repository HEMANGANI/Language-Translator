# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()

## From keras import 
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from keras.optimizers import RMSprop
#import tensorflow.python.keras.optimizers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

tf.compat.v1.enable_eager_execution()

#from google.colab import drive
#drive.mount('/content/gdrive', force_remount=True)

## Load the data
#file1 = open("gdrive/My Drive/Colab Notebooks/train_en.txt", encoding = "utf8")   # Load English Data
!wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en
file1 = open("/content/train.en", encoding="utf8")
english = file1.readlines()

#file2 = open("gdrive/My Drive/Colab Notebooks/train_vi.txt", encoding = "utf8")   # Load Vitnm Data
!wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi
file2 = open("/content/train.vi", encoding="utf8")
vitn = file2.readlines()

### Now add a start and end marker for the destination language. 
for i in range(0,len(vitn)):
    vitn[i] = "starttt " + vitn[i] + " enddd"

num_words = 10000 ## Most frequent 10,000 words for tokenizing. Make it 30k if dataset is large

class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""
    
    def __init__(self, texts, padding,
                 reverse=False, num_words=None):
        """
        :param texts: List of strings. This is the data-set.
        :param padding: Either 'post' or 'pre' padding.
        :param reverse: Boolean whether to reverse token-lists.
        :param num_words: Max number of words to use.
        """

        Tokenizer.__init__(self, num_words=num_words)
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))
        self.tokens = self.texts_to_sequences(texts)

        if reverse:
    
            self.tokens = [list(reversed(x)) for x in self.tokens]
            truncating = 'pre'
        else:
          
            truncating = 'post'

        self.num_tokens = [len(x) for x in self.tokens]

        self.max_tokens = np.mean(self.num_tokens) \
                          + 2 * np.std(self.num_tokens)
        self.max_tokens = int(self.max_tokens)
        self.tokens_padded = pad_sequences(self.tokens,
                                           maxlen=self.max_tokens,
                                           padding=padding,
                                           truncating=truncating)

    def token_to_word(self, token):

        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        
 
        text = " ".join(words)

        return text
    
    def text_to_tokens(self, text, reverse=False, padding=False):

        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)

        if reverse:
            tokens = np.flip(tokens, axis=1)

            truncating = 'pre'
        else:

            truncating = 'post'

        if padding:
            tokens = pad_sequences(tokens,
                                   maxlen=self.max_tokens,
                                   padding='pre',
                                   truncating=truncating)
        return tokens

### Now tokenize the datasets

tokenizer_eng = TokenizerWrap(texts=english,
                              padding='pre',
                              reverse=True,
                              num_words=num_words)     
 
tokenizer_vitn = TokenizerWrap(texts=vitn,
                               padding='post',
                               reverse=False,
                               num_words=num_words)

### This is to reduce the memory used for tokenizing the words.
tokens_eng = tokenizer_eng.tokens_padded
tokens_vitn = tokenizer_vitn.tokens_padded
print(tokens_eng.shape)
print(tokens_vitn.shape)
## Tokenizing done here ####

encoder_input_data = tokens_eng

decoder_input_data = tokens_vitn[:, :-1]       # They are in reverse format, so reverse them
decoder_output_data = tokens_vitn[:, 1:]       # First 'start' marker is time stepped in output

##### Neural Network #####

encoder_input = Input(shape=(None, ), name='encoder_input')

embedding_size = 128
state_size = 512

encoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='encoder_embedding')

encoder_gru1 = GRU(state_size, name='encoder_gru1',
                   return_sequences=True)
encoder_gru2 = GRU(state_size, name='encoder_gru2',
                   return_sequences=True)
encoder_gru3 = GRU(state_size, name='encoder_gru3',
                   return_sequences=False)

def connect_encoder():
    # Start the neural network with its input-layer.
    net = encoder_input
    
    # Connect the embedding-layer.
    net = encoder_embedding(net)

    # Connect all the GRU-layers.
    net = encoder_gru1(net)
    net = encoder_gru2(net)
    net = encoder_gru3(net)

    # This is the output of the encoder.
    encoder_output = net
    
    return encoder_output

## If we use LSTM's instead of GRU's at this place, We cannot take output the way we have taken now.     
encoder_output = connect_encoder()


######### Similarly build the decoder
decoder_initial_state = Input(shape=(state_size,),
                              name='decoder_initial_state')

decoder_input = Input(shape=(None, ), name='decoder_input')
decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')

decoder_gru1 = GRU(state_size, name='decoder_gru1',
                   return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',
                   return_sequences=True)

decoder_dense = Dense(num_words, activation='linear', name='decoder_output')

def connect_decoder(initial_state):
    # Start the decoder-network with its input-layer.
    net = decoder_input

    # Connect the embedding-layer.
    net = decoder_embedding(net)
    
    # Connect all the GRU-layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    # Connect the final dense layer that converts to
    # one-hot encoded arrays.
    decoder_output = decoder_dense(net)
    
    return decoder_output

######## Now connect all layers and create the model.
    
decoder_output = connect_decoder(initial_state=encoder_output)

model_train = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output])

model_encoder = Model(inputs=[encoder_input],
                      outputs=[encoder_output])

decoder_output = connect_decoder(initial_state=decoder_initial_state)

model_decoder = Model(inputs=[decoder_input, decoder_initial_state],
                      outputs=[decoder_output])

def sparse_cross_entropy(y_true, y_pred):
    """
    Calculate the cross-entropy loss between y_true and y_pred.
    
    y_true is a 2-rank tensor with the desired output.
    The shape is [batch_size, sequence_length] and it
    contains sequences of integer-tokens.

    y_pred is the decoder's output which is a 3-rank tensor
    with shape [batch_size, sequence_length, num_words]
    so that for each sequence in the batch there is a one-hot
    encoded array of length num_words.
    """

    # Calculate the loss. This outputs a
    # 2-rank tensor of shape [batch_size, sequence_length]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire 2-rank tensor, we reduce it
    # to a single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean

tf.executing_eagerly()

#### Compile the model

optimizer = RMSprop(lr=1e-3)    ### Adam / Adagrad dosent work well with RNN's
#decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
#model_train.compile(optimizer=optimizer, loss=sparse_cross_entropy, target_tensors=[decoder_target])
model_train.compile(optimizer=optimizer, loss=sparse_cross_entropy,run_eagerly=None)

### Callbacks
path_checkpoint = '21_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=3, verbose=1)
callback_tensorboard = TensorBoard(log_dir='./21_logs/',
                                   histogram_freq=0,
                                   write_graph=False)
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard]

####### Train the model

x_data = {
    'encoder_input': encoder_input_data,
    'decoder_input': decoder_input_data
}

y_data = {
    'decoder_output': decoder_output_data
}

validation_split = 10000 / len(encoder_input_data)
print (validation_split)
model_train.fit(x=x_data,
                y=y_data,
                batch_size=512,
                epochs=4,
                validation_split=validation_split,
                callbacks = callbacks)

def train():
    validation_split = 10000 / len(encoder_input_data)
    print (validation_split)

    model_train.fit(x=x_data,
                    y=y_data,
                    batch_size=512,
                    epochs=4,
                    validation_split=validation_split,
                    callbacks = callbacks)

mark_start = 'starttt'
mark_end = 'enddd'
token_start = tokenizer_vitn.word_index[mark_start.strip()]
token_end = tokenizer_vitn.word_index[mark_end.strip()]

##### Save the trained model in Colab
model_train.save('training_model.h5')
"""
model_train.save('training_model.h5')
from google.colab import files
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# create on Colab directory
model_train.save('training_model.h5')    
model_file = drive.CreateFile({'title' : 'training_model.h5'})
model_file.SetContentFile('training_model.h5')
model_file.Upload()

# download to google drive
drive.CreateFile({'id': model_file.get('id')})
"""

def translate(input_text, true_output_text=None):
    """Translate a single text-string."""

    # Convert the input-text to integer-tokens.
    # Note the sequence of tokens has to be reversed.
    # Padding is probably not necessary.
    input_tokens = tokenizer_eng.text_to_tokens(text=input_text,
                                                reverse=True,
                                                padding=True)
    
    # Get the output of the encoder's GRU which will be
    # used as the initial state in the decoder's GRU.
    # This could also have been the encoder's final state
    # but that is really only necessary if the encoder
    # and decoder use the LSTM instead of GRU because
    # the LSTM has two internal states.
    initial_state = model_encoder.predict(input_tokens)

    # Max number of tokens / words in the output sequence.
    max_tokens = tokenizer_vitn.max_tokens

    # Pre-allocate the 2-dim array used as input to the decoder.
    # This holds just a single sequence of integer-tokens,
    # but the decoder-model expects a batch of sequences.
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    # The first input-token is the special start-token for 'ssss '.
    token_int = token_start

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0
    
    # While we haven't sampled the special end-token for ' eeee'
    # and we haven't processed the max number of tokens.
    while token_int != token_end and count_tokens < max_tokens:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = \
        {
            'decoder_initial_state': initial_state,
            'decoder_input': decoder_input_data
        }

        # Note that we input the entire sequence of tokens
        # to the decoder. This wastes a lot of computation
        # because we are only interested in the last input
        # and output. We could modify the code to return
        # the GRU-states when calling predict() and then
        # feeding these GRU-states as well the next time
        # we call predict(), but it would make the code
        # much more complicated.

        # Input this data to the decoder and get the predicted output.
        decoder_output = model_decoder.predict(x_data)

        # Get the last predicted token as a one-hot encoded array.
        token_onehot = decoder_output[0, count_tokens, :]
        
        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer_vitn.token_to_word(token_int)

        # Append the word to the output-text.
        output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    # Sequence of tokens output by the decoder.
    output_tokens = decoder_input_data[0]
     # Print the input-text.
    print("Input text:")
    print(input_text)
    print()

    # Print the translated output-text.
    print("Translated text:")
    print(output_text)
    print()

    # Optionally print the true translated text.
    if true_output_text is not None:
        print("True output text:")
        print(true_output_text)
        print()
    
    return input_text, output_text, true_output_text

idx = 7
input_text, output_text, true_output_text = translate(input_text=english[idx],
         true_output_text=vitn[idx])

translate(input_text = 'This was an amazing day')

translate(input_text = 'This is a cat')

translate(input_text = 'Our college is in trichy')

def translate1(input_text, true_output_text=None):
    """Translate a single text-string."""

    # Convert the input-text to integer-tokens.
    # Note the sequence of tokens has to be reversed.
    # Padding is probably not necessary.
    input_tokens = tokenizer_eng.text_to_tokens(text=input_text,
                                                reverse=True,
                                                padding=True)
    
    # Get the output of the encoder's GRU which will be
    # used as the initial state in the decoder's GRU.
    # This could also have been the encoder's final state
    # but that is really only necessary if the encoder
    # and decoder use the LSTM instead of GRU because
    # the LSTM has two internal states.
    initial_state = model_encoder.predict(input_tokens)

    # Max number of tokens / words in the output sequence.
    max_tokens = tokenizer_vitn.max_tokens

    # Pre-allocate the 2-dim array used as input to the decoder.
    # This holds just a single sequence of integer-tokens,
    # but the decoder-model expects a batch of sequences.
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    # The first input-token is the special start-token for 'ssss '.
    token_int = token_start

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0
    
    # While we haven't sampled the special end-token for ' eeee'
    # and we haven't processed the max number of tokens.
    while token_int != token_end and count_tokens < max_tokens:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = \
        {
            'decoder_initial_state': initial_state,
            'decoder_input': decoder_input_data
        }

        # Note that we input the entire sequence of tokens
        # to the decoder. This wastes a lot of computation
        # because we are only interested in the last input
        # and output. We could modify the code to return
        # the GRU-states when calling predict() and then
        # feeding these GRU-states as well the next time
        # we call predict(), but it would make the code
        # much more complicated.

        # Input this data to the decoder and get the predicted output.
        decoder_output = model_decoder.predict(x_data)

        # Get the last predicted token as a one-hot encoded array.
        token_onehot = decoder_output[0, count_tokens, :]
        
        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer_vitn.token_to_word(token_int)

        # Append the word to the output-text.
        output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    # Sequence of tokens output by the decoder.
    output_tokens = decoder_input_data[0]
    # Print the input-text
    
    return input_text, output_text, true_output_text

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
sm = SmoothingFunction()
score = sentence_bleu([output_text], true_output_text, smoothing_function=sm.method1)
print(score)

"""
def test():
    ### Load the data
    file1 = open("gdrive/My Drive/Colab Notebooks/tst2012_en.txt", encoding = "utf8")   # Load English Data
    english_test = file1.readlines()

    file2 = open("gdrive/My Drive/Colab Notebooks/tst2012_vi.txt", encoding = "utf8")   # Load Vitnm Data
    vitn_test = file2.readlines()

    ### Now add a start and end marker for the destination language. 
    for i in range(0,len(vitn_test)):
        vitn_test[i] = "starttt " + vitn_test[i] + " enddd"
    
    #count = 0
    scores_list = []
    for idx in range(0,20): # Doing for 100 lines
        input_text, output_text, true_output_text = translate1(input_text=english_test[idx],true_output_text=vitn_test[idx])
        scor = sentence_bleu([output_text], true_output_text, smoothing_function=sm.method1)
        scores_list.append(scor)
        #print(scor)
        
    BLEU_average = sum(scores_list)/ 20
    print ("The BLEU average score for the test_data = ", BLEU_average)
    #print(count)
    
    return BLEU_average
   """

#BLEU_average = test()
