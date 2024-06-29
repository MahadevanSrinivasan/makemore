import tensorflow as tf
from tensorflow import keras

class MLPBigram:
  def __init__(self, words):
    # Create the training set of all bigrams
    chars = sorted(list(set(''.join(words))))
    vocab = chars + ['.']
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    self.itos_ = {i:s for s,i in stoi.items()}
    
    xs, ys = [], []
    for w in words:
      chs = ['.'] + list(w) + ['.']
      for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(ch1)
        ys.append(ch2)
    
    # Can't assign to tf tensor values
    xs = tf.constant(xs)
    ys = tf.constant(ys)
    
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    preprocessing_layer = tf.keras.layers.StringLookup(max_tokens=27, vocabulary=vocab, num_oov_indices=0, output_mode='one_hot')
    x = preprocessing_layer(inputs)
    outputs = tf.keras.layers.Dense(27, activation='softmax', kernel_initializer='zeros')(x)
    self.model_ = tf.keras.Model(inputs, outputs)

    self.label_preprocessing_layer_ = tf.keras.layers.StringLookup(max_tokens=27, vocabulary=vocab, num_oov_indices=0, output_mode='int')
    ys_enc = self.label_preprocessing_layer_(ys)

    self.output_post_process_layer_ = tf.keras.layers.StringLookup(vocabulary=vocab, invert=True)

    self.model_.compile(optimizer=keras.optimizers.Adam(learning_rate=1.0), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    history = self.model_.fit(xs, ys_enc, epochs=10, batch_size=len(ys), verbose=0)
    # plt.plot(history.history['loss'])
    print(history.history['loss'][-1])

  def __call__(self, x: str):
    p = self.model_(tf.constant([x]))
    ix = tf.random.categorical(p, num_samples=1)[0,0].numpy()
    result = self.itos_[ix]
    return result

  def generate(self):
    prev_token = '.'
    results = []
    while True:
        next_token = self(prev_token)
        if next_token == '.':
            break
        results.append(next_token)
        prev_token = next_token
    return ''.join(results)

if __name__ == '__main__':
  words = open('names.txt', 'r').read().splitlines()
  tf.random.set_seed(42)
  model = MLPBigram(words)
  for i in range(5):
    print(model.generate())