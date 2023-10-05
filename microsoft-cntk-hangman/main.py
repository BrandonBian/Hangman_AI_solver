# %% [markdown]
# # Train a Neural Network to Play Hangman
# 
# by Mary Wahl, Shaheen Gauher, Fidan Boylu Uz, Katherine Zhao
# 
# In the classic children's game of Hangman, a player's objective is to identify a hidden word of which only the number of letters is originally revealed. In each round, the player guesses a letter of the alphabet: if it's present in the word, all instances are revealed; otherwise one of the hangman's body parts is drawn in on a gibbet. The game ends in a win if the word is entirely revealed by correct guesses, and ends in loss if the hangman's body is completely revealed instead. To assist the player, a visible record of all guessed letters is typically maintained.
# 
# The goal of this project was to use reinforcement learning to train a neural network to play Hangman by appropriately guessing letters in a partially or fully obscured word. The network receives as input a representation of the word (total number of characters, the identity of any revealed letters) as well as a list of which letters have been guessed so far. It returns a guess for the letter that should be picked next. This notebook shows our method for training the network and validating its performance on a withheld test set.

# %% [markdown]
# ## Outline
# 
# - [Set up the execution environment](#setup)
# - [Extract a list of unique words from the input data](#input)
# - [Partition the words into training and validation sets](#split)
# - [Create the game player](#player)
# - [Create the model](#model)
# - [Train the model](#train)
# - [Evaluating results](#eval)
# - [Just for fun -- play hangman with your favorite word](#fun)

# %% [markdown]
# <a name="setup"></a>
# ## Set up the execution environment
# 
# There are two readily available options on Azure to run this notebook if you should choose to do so.
# 
# 1) Create a GPU VM with CNTK 2.0 RC2 pre-installed using the [Azure Deep Learning toolkit for the DSVM](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.dsvm-deep-learning) and set up the VM's Jupyter Notebook server using the [provided instructions](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-provision-vm#how-to-create-a-strong-password-for-jupyter-and-start-the-notebook-server).
# 
# 2) Provision a [Microsoft Azure Data Science Virtual Machine (DSVM)](https://blogs.technet.microsoft.com/machinelearning/2017/06/06/introducing-the-new-data-science-virtual-machine-on-windows-server-2016/) with Windows Server 2016. They come  pre-installed with the GPU Nvidia drivers, CUDA toolkit 8.0, and cuDNN library.
# 
# We then loaded this notebook on the VM before executing the code cells below. VM images are updated regularly, but at the time of this writing, the following package versions were pre-installed in the `py35` Anaconda environment on the VMs respectively:

# %% [markdown]
# The notebook was tested with   
# Python version: 3.5.2 and 3.5.3  
# Anaconda 4.4.0 (64-bit)  
# CNTK version: 2.0rc2 and 2.0  
# NumPy version: 1.11.2 and 1.13.0  
# Pandas version: 0.19.1 and 0.20.0  

# %%
import sys, cntk
import numpy as np
import pandas as pd

print('''
Python version: {}
CNTK version: {}
NumPy version: {}
Pandas version: {}
'''.format(sys.version, cntk.__version__, np.__version__, pd.__version__))

# %% [markdown]
# If your output is different when you run the code cells above, version differences may impact this notebook's function. You can use the following command to install a specific package version if necessary:
# 
# `pip install <package-name>==<version-number> --force-reinstall`

# %% [markdown]
# <a name="input"></a>
# ## Extract a list of unique words from the input data
# 
# We train and validate our model using words from Princeton University's [WordNet](http://wordnet.princeton.edu) database. Specifically, we download the [tarballed version of WordNet 3.0](http://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz) and use all words (consisting only of alphabetic characters) from the following files from the tarball's `dict` subfolder (which we transfer to a local folder named `input_data`):
# - `data.adj`
# - `data.adv`
# - `data.verb`
# - `data.noun`

# %% [markdown]
# We augment this word set with a few phrases, like "Microsoft" and "CNTK". We then randomly partition the words 80:20 to create training and validation sets of 44,503 and 11,226 words, respectively.

# %%
# word_dict = {}
# input_files = ['./input_data/data.adj',
#                './input_data/data.adv',
#                './input_data/data.noun',
#                './input_data/data.verb']

# for filename in input_files:
#     with open(filename, 'r') as f:
#         # skip the header lines
#         for i in range(29):
#             f.readline()

#         for line in f:
#             word = line.split(' ')[4]
#             if word.isalpha():
#                 word_dict[word.lower()] = None

# word_dict['microsoft'] = None
# word_dict['cntk'] = None

# # create a list to be used as input later
# words = list(np.random.permutation(list(word_dict.keys())))
# with open('word_list.txt', 'w') as f:
#     for word in words:
#         f.write('{}\n'.format(word))

# %%
# Step 1: Read words from a text file
file_path = './words_250000_train.txt'
with open(file_path, 'r') as file:
    words = file.read().splitlines()

# Step 2: Create a dictionary with words as keys and None as values
word_dict = {word: None for word in words}

# Print the dictionary to verify
print(len(word_dict))

# %% [markdown]
# <a name="split"></a>
# ## Partition the words into training and validation sets

# %%
# During training, the model will only see words below this index.
# The remainder of the words can be used as a validation set.
train_val_split_idx = int(len(list(word_dict.keys())) * 0.8)
print('Training with {} WordNet words'.format(train_val_split_idx))

MAX_NUM_INPUTS = max([len(i) for i in words[:train_val_split_idx]])
EPOCH_SIZE = train_val_split_idx
BATCH_SIZE = np.array([len(i) for i in words[:train_val_split_idx]]).mean()
print('Max word length: {}, average word length: {:0.1f}'.format(MAX_NUM_INPUTS, BATCH_SIZE))
print("Epoch Size:", EPOCH_SIZE)
print("Batch Size:", BATCH_SIZE)

# %% [markdown]
# <a name="player"></a>
# ## Create the game player
# 
# This is a "wrapper" of sorts around our neural network model, that handles the dynamics of gameplay.

# %%
class HangmanPlayer:
    def __init__(self, word, model, lives=6): # Set to equal to number of lifes for the API
        self.original_word = word
        self.full_word = [ord(i)-97 for i in word]
        self.letters_guessed = set([])
        self.letters_remaining = set(self.full_word)
        self.lives_left = lives
        self.obscured_words_seen = []
        self.letters_previously_guessed = []
        self.guesses = []
        self.correct_responses = []
        self.z = model
        return
    
    def encode_obscured_word(self):
        word = [i if i in self.letters_guessed else 26 for i in self.full_word]
        obscured_word = np.zeros((len(word), 27), dtype=np.float32)
        for i, j in enumerate(word):
            obscured_word[i, j] = 1
        return(obscured_word)
    
    def encode_guess(self, guess):
        encoded_guess = np.zeros(26, dtype=np.float32)
        encoded_guess[guess] = 1
        return(encoded_guess)

    def encode_previous_guesses(self):
        # Create a 1 x 26 vector where 1s indicate that the letter was previously guessed
        guess = np.zeros(26, dtype=np.float32)
        for i in self.letters_guessed:
            guess[i] = 1
        return(guess)
    
    def encode_correct_responses(self):
        # To be used with cross_entropy_with_softmax, this vector must be normalized
        response = np.zeros(26, dtype=np.float32)
        for i in self.letters_remaining:
            response[i] = 1.0
        response /= response.sum()
        return(response)
    
    def store_guess_and_result(self, guess):
        # Record what the model saw as input: an obscured word and a list of previously-guessed letters
        self.obscured_words_seen.append(self.encode_obscured_word())
        self.letters_previously_guessed.append(self.encode_previous_guesses())
        
        # Record the letter that the model guessed, and add that guess to the list of previous guesses
        self.guesses.append(guess)
        self.letters_guessed.add(guess)
        
        # Store the "correct responses"
        correct_responses = self.encode_correct_responses()
        self.correct_responses.append(correct_responses)
        
        # Determine an appropriate reward, and reduce # of lives left if appropriate
        if guess in self.letters_remaining:
            self.letters_remaining.remove(guess)
        
        if self.correct_responses[-1][guess] < 0.00001:
            self.lives_left -= 1
        return
                
    def run(self):
        # Play a game until we run out of lives or letters
        while (self.lives_left > 0) and (len(self.letters_remaining) > 0):
            guess = np.argmax(np.squeeze(self.z.eval({self.z.arguments[0]: np.array(self.encode_obscured_word()),
                                                      self.z.arguments[1]: np.array(self.encode_previous_guesses())})))
            self.store_guess_and_result(guess)
        
        # Return the observations for use in training (both inputs, predictions, and losses)
        return(np.array(self.obscured_words_seen),
               np.array(self.letters_previously_guessed),
               np.array(self.correct_responses))
    
    def show_words_seen(self):
        for word in self.obscured_words_seen:
            print(''.join([chr(i + 97) if i != 26 else ' ' for i in word.argmax(axis=1)]))
            
    def show_guesses(self):
        for guess in self.guesses:
            print(chr(guess + 97))
            
    def play_by_play(self):
        print('Hidden word was "{}"'.format(self.original_word))
        for i in range(len(self.guesses)):
            word_seen = ''.join([chr(i + 97) if i != 26 else ' ' for i in self.obscured_words_seen[i].argmax(axis=1)])
            print('Guessed {} after seeing "{}"'.format(chr(self.guesses[i] + 97),
                                                        word_seen))
            
    def evaluate_performance(self):
        # Assumes that the run() method has already been called
        ended_in_success = self.lives_left > 0
        letters_in_word = set([i for i in self.original_word])
        correct_guesses = len(letters_in_word) - len(self.letters_remaining)
        incorrect_guesses = len(self.guesses) - correct_guesses
        return(ended_in_success, correct_guesses, incorrect_guesses, letters_in_word)

# %% [markdown]
# <a name="model"></a>
# ## Create the model
# 
# The network will accept as input:
# - an n-letter obscured word, with letters/blanks encoded as one-hots (n x 27 dense vector)
# - a 1 x 26 vector of guesses made so far (1 if the letter has been guessed; 0 otherwise
# 
# It will return as output:
# - a 1 x 26 vector of which argmax is the letter the model "chooses next"
# 
# The variable-length obscured word is fed into an LSTM, and the final output of the LSTM is combined with info on the guesses so far before entering dense layers.

# %%
def create_LSTM_net(input_obscured_word_seen, input_letters_guessed_previously):
    with cntk.layers.default_options(initial_state = 0.1):
        lstm_outputs = cntk.layers.Recurrence(cntk.layers.LSTM(MAX_NUM_INPUTS))(input_obscured_word_seen)
        final_lstm_output = cntk.ops.sequence.last(lstm_outputs)
        combined_input = cntk.ops.splice(final_lstm_output, input_letters_guessed_previously)
        dense_layer = cntk.layers.Dense(26, name='final_dense_layer')(combined_input)
        return(dense_layer)
    
input_obscured_word_seen = cntk.ops.input_variable(shape=27,
                                                   dynamic_axes=[cntk.Axis.default_batch_axis(),
                                                                 cntk.Axis.default_dynamic_axis()],
                                                   name='input_obscured_word_seen')
input_letters_guessed_previously = cntk.ops.input_variable(shape=26,
                                                           dynamic_axes=[cntk.Axis.default_batch_axis()],
                                                           name='input_letters_guessed_previously')

z = create_LSTM_net(input_obscured_word_seen, input_letters_guessed_previously)

# %% [markdown]
# <a name="train"></a>
# ## Train the model
# 
# Set some learning parameters:

# %%
# define loss and displayed metric
input_correct_responses = cntk.ops.input_variable(shape=26,
                                                  dynamic_axes=[cntk.Axis.default_batch_axis()],
                                                  name='input_correct_responses')
pe = cntk.losses.cross_entropy_with_softmax(z, input_correct_responses)
ce = cntk.metrics.classification_error(z, input_correct_responses)

learning_rate = 0.1
lr_schedule = cntk.learners.learning_rate_schedule(learning_rate, cntk.UnitType.minibatch)
momentum_time_constant = cntk.learners.momentum_as_time_constant_schedule(BATCH_SIZE / -np.log(0.9)) 
learner = cntk.learners.fsadagrad(z.parameters,
                                  lr=lr_schedule,
                                  momentum=momentum_time_constant,
                                  unit_gain = True)
trainer = cntk.Trainer(z, (pe, ce), learner)
progress_printer = cntk.logging.progress_print.ProgressPrinter(freq=EPOCH_SIZE, tag='Training')

# %% [markdown]
# Perform the actual training using the code cell below. Note that this step will take many hours to complete:

# %%
from tqdm import tqdm

NUM_EPOCHS = 20
total_samples = 0
model_filename = './final_model.dnn'

for epoch in range(NUM_EPOCHS):
    i = 0
    print("--- Epoch " + str(i) + " ---")
    pbar = tqdm(total=(epoch+1) * EPOCH_SIZE, dynamic_ncols=True)

    while total_samples < (epoch+1) * EPOCH_SIZE:
        word = words[i]
        i += 1
        
        other_player = HangmanPlayer(word, z)
        words_seen, previous_letters, correct_responses = other_player.run()
        
        trainer.train_minibatch({input_obscured_word_seen: words_seen,
                                 input_letters_guessed_previously: previous_letters,
                                 input_correct_responses: correct_responses})

        # Output progress
        # loss_avg = round(trainer.previous_minibatch_loss_average, 4)
        # sample_cnt = trainer.previous_minibatch_sample_count
        # eval_avg = round(trainer.previous_minibatch_evaluation_average, 4)
        
        # update_string = "loss_avg: " + str(loss_avg) + " | sample_cnt: " + str(sample_cnt) + " | eval_avg: " + str(eval_avg)

        # Update the tqdm progress bar description with the string
        # pbar.set_description(update_string)
        pbar.update(1)  # Update the progress bar

        total_samples += 1
        progress_printer.update_with_trainer(trainer, with_metric=True)
    
    # Close the tqdm progress bar for the epoch
    pbar.close()

    # Save per epoch
    print("Saving model...")
    z.save(model_filename)
        
    progress_printer.epoch_summary(with_metric=True)

# %% [markdown]
# When interpreting the loss and metric during training, keep in mind:
# - The "classification error" is the $\ell_1$ distance between the softmax of the model's output and a normalized vector indicating the true responses. For example, if the letters "a" and "b" were the only correct responses, the normalized vector would be [0.5, 0.5, 0, ... 0]. It is *not* the fraction of incorrect guesses and not easily interpreted in an absolute sense. (We will switch to human-interpretable metrics after training finishes; for now, look for improvements in performance.)
# - The loss function is the cross entropy between the softmax of the model's output and the normalized vector described above.
# 
# Expect training to take several hours on an Azure NC6 GPU DSVM.
# 
# ### Save the model

# %%
# model_filename = './hangman_model_sample.dnn'
# z.save(model_filename)

# %% [markdown]
# <a name="eval"></a>
# ## Evaluating results
# 
# ### Anecdotal version of performance evaluation
# Let's check out how the game went for the last word seen during training. (That game's results are still stored in the variable `other_player`.) Your word will of course vary due to random word shuffling during data partitioning.

# %%
other_player.play_by_play()

# %% [markdown]
# In this specific case, the performance is encouraging!
# 
# ### More thorough version of performance evaluation
# 
# Now we play hangman with all words in the validation set, and quantify performance with a few metrics:
# - fraction of games won
# - average number of correct guesses
# - average number of incorrect guesses
# 
# Note that the number of incorrect guesses is bounded by the number of lives per game (set to "10" as of this writing), i.e. how many "body parts" one draws on the hangman.
# 
# First we evaluate the model on all words in the validation set:

# %%
from tqdm import tqdm

model_filename = './hangman_model_ref.dnn'
current_model = cntk.load_model(model_filename)

# %%
def evaluate_model(my_words, my_model):
    results = []
    for word in tqdm(my_words):
        my_player = HangmanPlayer(word, my_model)
        _ = my_player.run()
        results.append(my_player.evaluate_performance())
    df = pd.DataFrame(results, columns=['won', 'num_correct', 'num_incorrect', 'letters'])
    return(df)

# Expect this to take roughly ten minutes
result_df = evaluate_model(words[train_val_split_idx:], current_model)

# %% [markdown]
# Then we summarize the results:

# %%
print('Performance on the validation set:')
print('- Averaged {:0.1f} correct and {:0.1f} incorrect guesses per game'.format(result_df['num_correct'].mean(),
                                                                       result_df['num_incorrect'].mean()))
print('- Won {:0.1f}% of games played'.format(100 * result_df['won'].sum() / len(result_df.index)))

# %% [markdown]
# <a name="fun"></a>
# ## Just for fun -- play hangman with your favorite word
# 
# In case you would like to see how the trained model performs on your favorite word!

# %%
model_filename = './hangman_model_ref.dnn'
z2 = cntk.load_model(model_filename)
my_word = 'microsoft'

my_player = HangmanPlayer(my_word, z2)
_ = my_player.run()
my_player.play_by_play()

# %%
results = my_player.evaluate_performance()
print('The model {} this game'.format('won' if results[0] else 'did not win'))
print('The model made {} correct guesses and {} incorrect guesses'.format(results[1], results[2]))

# %% [markdown]
# ## API

# %%
word = "app_e"

# %%
# Encode and mask guessed letters
guessed_letters = ['x', 'y', 'z']

encode_previous_guesses = np.zeros(26, dtype=np.float32)
for letter in guessed_letters:
    # Already guessed, mask it
    encode_previous_guesses[ord(letter) - ord('a')] = 1

print(encode_previous_guesses)

# %%
def obscure(word):
    # Ref (Azure Tutorial on LSTM Obscuring): https://github.com/Azure/Hangman/tree/master
    word_idx = [ord(i) - ord('a') if i != "_" else 26 for i in word]
    obscured_word = np.zeros((len(word), 27), dtype=np.float32)
    for i, j in enumerate(word_idx):
        obscured_word[i, j] = 1
        
    return obscured_word

obscure(word)

# %%
def encode_obscured_word(word):
    word = [i if i in guessed_letters else 26 for i in word]
    obscured_word = np.zeros((len(word), 27), dtype=np.float32)
    for i, j in enumerate(word):
        obscured_word[i, j] = 1
    return(obscured_word)

encode_obscured_word(word)

# %%
# Inference
prediction = np.argmax(np.squeeze(current_model.eval(
    ({current_model.arguments[0]: np.array(obscure(word)),
      current_model.arguments[1]: np.array(encode_previous_guesses)})
)))

guess = chr(prediction + ord('a'))

guess

# %%
current_model = cntk.load_model("././hangman_model_ref.dnn")
my_player = HangmanPlayer("gamma", current_model)
_ = my_player.run()
my_player.play_by_play()

# %%



