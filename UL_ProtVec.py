from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
from matplotlib import pylab
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from sklearn.manifold import TSNE
from scipy.sparse import lil_matrix

"""
This script borrows inspiration from:
https://github.com/wangz10/tensorflow-playground
"""


# File names for different data; A - alpha chain, B - beta chain            
cd4A_file = 'patient1/vDCRe_alpha_EG10_CD4_naive_alpha.txt'
cd8A_file = 'patient1/vDCRe_alpha_EG10_CD8_naive_alpha.txt'
cd4B_file = 'patient1/vDCRe_beta_EG10_CD4_naive_beta.txt'
cd8B_file = 'patient1/vDCRe_beta_EG10_CD8_naive_beta.txt'
data = 'data/'
extra = 'extra/'
########################################################
# Data Retrieval 
########################################################
"""
The data for the first patient is stored in the 'data/patient1' file
where each sequence is encoded as a string and comma separated with its count
at current extraction is unique and count is ignored
"""
# Files to be read
files = [cd4A_file, cd8A_file]

# sequence list to be filled. Make sure file order is the same as a below
cd4=[]
cd8=[]
seqs=[cd4,cd8]

# if the data files you want only have cdr3s it has to parse it differently
for index, file in enumerate(files):
    file=data+file
    with open(file,'r') as infile:
        # goes through each of the files specified in read mode and pulls out 
        # each line and formats it so a list gets X copies of the sequence 
        for line in infile:
            twoVals=line.split(", ")
            twoVals[1]=twoVals[1].replace("\n","")
            for i in range(int(twoVals[1])):
                seqs[index].append(twoVals[0])

# combine the sequence lists together
# they havent been split into tuples yet
seqs=cd4+cd8

# empty list for new seqs 
tupSeqs=[]

# function for pTuples
def pTuple(vec,n=3):
    """Returns a vector of ptuples from a given sequence"""
    return [vec[i:i+n] for i in range(len(vec)-n+1)]

# go through each sequence and split the tuples into separate non overlappting seqs
n=4
print("{} Sequences".format(len(seqs)))
for idx, seq in enumerate(seqs):
    seq = pTuple(seq,n=n)
    short=[]
    for i in range(n):
        for j in range(len(seq)):
            try:
                short.append(seq[i+(n*j)])
            except:
                break
    tupSeqs.append(short)
    if idx % 100 == 0:
        print("{}/{} Sequences Read".format(idx+1, len(seqs)))
seqs=None
words = tupSeqs
# flattens list
words = [item for sublist in words for item in sublist]    


# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 40000

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.


data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  weights = np.ndarray(shape=(batch_size), dtype=np.float32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
      weights[i * num_skips + j] = abs(1.0/(target - skip_window))
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels, weights

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (8, 4)]:
    data_index = 0
    batch, labels, weights = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
    print('    weights:', [w for w in weights])
    

    
    
# We are creating the co-occurance matrix as a compressed sparse colum matrix from scipy. 
# initially I was thinking of using SparseTensor (from tensorflow) however it's not well-devloped (cannot change individual elements)
cooc_data_index = 0
dataset_size = len(data)
skip_window = 4 # How many words to consider left and right.
num_skips = 8 # How many times to reuse an input to generate a label.
#dataset_size = 1000000 #used the value for test purposes
cooc_mat = lil_matrix((vocabulary_size, vocabulary_size), dtype=np.float32)
print(cooc_mat.shape)
def generate_cooc(batch_size,num_skips,skip_window):
    data_index = 0
    print('Running %d iterations to compute the co-occurance matrix'%(dataset_size//batch_size))
    for i in range(dataset_size//batch_size):
        if i>0 and i%100000==0:
            print('\tFinished %d iterations'%i)
        batch, labels, weights = generate_batch(batch_size=batch_size, num_skips=num_skips, skip_window=skip_window) # increments data_index automatically
        labels = labels.reshape(-1)
        
        for inp,lbl,w in zip(batch,labels,weights):            
            cooc_mat[inp,lbl] += (1.0*w)
            
generate_cooc(8,num_skips,skip_window)    

# just printing some parts of co-occurance matrix
print('Sample chunks of co-occurance matrix')
rand_target_idx = np.random.randint(0,vocabulary_size,10).tolist()
for i in range(10):
    idx_target = i
    ith_row = cooc_mat.getrow(idx_target) # get the ith row of the sparse matrix
    # couldn't use todense() but toarray() works
    # need to find the difference (welcome any reasoning behind this)
    ith_row_dense = ith_row.toarray('C').reshape(-1)        
    # select target words only with a reasonable words around it.
    while np.sum(ith_row_dense)<10 or np.sum(ith_row_dense)>50000:
        idx_target = np.random.randint(0,vocabulary_size)
        ith_row = cooc_mat.getrow(idx_target) # get the ith row of the sparse matrix
        ith_row_dense = ith_row.toarray('C').reshape(-1)    
        
    print('\nTarget Word: "%s"'%reverse_dictionary[idx_target])
        
    sort_indices = np.argsort(ith_row_dense).reshape(-1) # indices with highest count of ith_row_dense
    sort_indices = np.flip(sort_indices,axis=0) # reverse the array (to get max values to the start)

    # printing several context words to make sure cooc_mat is correct
    print('Context word:',end='')
    for j in range(10):        
        idx_context = sort_indices[j]       
        print('"%s"(id:%d,count:%.2f), '%(reverse_dictionary[idx_context],idx_context,ith_row_dense[idx_context]),end='')
    print()
    
    
batch_size = 128
embedding_size = 150 # Dimension of the embedding vector.

# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. 
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.

# Validation set consist of 50 infrequent words and 50 frequent words
valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
valid_examples = np.append(valid_examples,random.sample(range(1000,1000+valid_window), valid_size//2))

epsilon = 1 # used for the stability of log in the loss function
graph = tf.Graph()

#with graph.as_default(), tf.device('/cpu:0'):

# Input data.
train_dataset = tf.placeholder(tf.int32, shape=[batch_size],name='train_dataset')
train_labels = tf.placeholder(tf.int32, shape=[batch_size],name='train_labels')
valid_dataset = tf.constant(valid_examples, dtype=tf.int32,name='valid_dataset')

# Variables.
embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),name='embeddings')
bias_embeddings = tf.Variable(tf.random_uniform([vocabulary_size],0.0,0.01,dtype=tf.float32),name='embeddings_bias')

# Model.
# Look up embeddings for inputs.
embed_in = tf.nn.embedding_lookup(embeddings, train_dataset)
embed_out = tf.nn.embedding_lookup(embeddings, train_labels)
embed_bias_in = tf.nn.embedding_lookup(bias_embeddings,train_dataset)
embed_bias_out = tf.nn.embedding_lookup(bias_embeddings,train_labels)

# weights used in the cost function
weights_x = tf.placeholder(tf.float32,shape=[batch_size],name='weights_x') 
x_ij = tf.placeholder(tf.float32,shape=[batch_size],name='x_ij')

# Compute the loss defined in the paper. Note that I'm not following the exact equation given (which is computing a pair of words at a time)
# I'm calculating the loss for a batch at one time, but the calculations are identical.
# I also made an assumption about the bias, that it is a smaller type of embedding
loss = tf.reduce_mean(
    weights_x * (tf.reduce_sum(embed_in*embed_out,axis=1) + embed_bias_in + embed_bias_out - tf.log(epsilon+x_ij))**2)

# Optimizer.
# Note: The optimizer will optimize the softmax_weights AND the embeddings.
# This is because the embeddings are defined as a variable quantity and the
# optimizer's `minimize` method will by default modify all variable quantities 
# that contribute to the tensor it is passed.
# See docs on `tf.train.Optimizer.minimize()` for more details.
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

# Compute the similarity between minibatch examples and all embeddings.
# We use the cosine distance:
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(
normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
    




num_steps = 100001

session = tf.InteractiveSession()

tf.global_variables_initializer().run()

print('Initialized')
average_loss = 0
for step in range(num_steps):
    batch_data, batch_labels, batch_weights = generate_batch(
        batch_size, num_skips, skip_window) # generate a single batch (data,labels,co-occurance weights)
    batch_weights = [] # weighting used in the loss function
    batch_xij = [] # weighted frequency of finding i near j
    for inp,lbl in zip(batch_data,batch_labels.reshape(-1)):        
        batch_weights.append((np.asscalar(cooc_mat[inp,lbl])/100.0)**0.75)
        batch_xij.append(cooc_mat[inp,lbl])
    batch_weights = np.clip(batch_weights,-100,1)
    batch_xij = np.asarray(batch_xij)

    feed_dict = {train_dataset : batch_data.reshape(-1), train_labels : batch_labels.reshape(-1),
                weights_x:batch_weights,x_ij:batch_xij}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)

    average_loss += l
    if step % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step %d: %f' % (step, average_loss))
      average_loss = 0
    # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        print(log)
final_embeddings = normalized_embeddings.eval()

np.save("embeddings/embed",final_embeddings)
np.save("embeddings/revDict",reverse_dictionary)
np.save("embeddings/dict_norm",dictionary)

# tSNE
num_points = 10000
offset = 1000
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#two_d_embeddings_1 = tsne.fit_transform(final_embeddings[1:(num_points//2)+1,:])
#two_d_embeddings_2 = tsne.fit_transform(final_embeddings[offset:offset+(num_points//2), :])
two_d_embeddings_3 = tsne.fit_transform(final_embeddings)

def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    #pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
    #               ha='right', va='bottom')
  pylab.show()

words_1 = [reverse_dictionary[i] for i in range(1, (num_points//2)+1)]
words_2 = [reverse_dictionary[i] for i in range(offset, offset + (num_points//2))]

#plot(two_d_embeddings_1, words_1)
#plot(two_d_embeddings_2, words_2)
plot(two_d_embeddings_3, np.zeros((two_d_embeddings_3.shape[0])))

