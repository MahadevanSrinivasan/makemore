from typing import List
import tensorflow as tf

class BigramModel:
    def __init__(self, words : List[str]):
        chars = sorted(list(set(''.join(words))))
        self.counts_ = [[0] * (len(chars)+1) for _ in range(len(chars)+1)]
        self.stoi_ = {s:i+1 for i,s in enumerate(chars)}
        self.stoi_['.'] = 0
        self.itos_ = {i:s for s,i in self.stoi_.items()}
        for w in words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.stoi_[ch1]
                ix2 = self.stoi_[ch2]
                self.counts_[ix1][ix2] += 1
        self.P_ = tf.constant(self.counts_, dtype=tf.float32)
        self.P_ /= tf.math.reduce_sum(self.P_, axis=1, keepdims=True)
    
    def __call__(self, x: str):
        ix = self.stoi_[x]
        p = self.P_[ix].numpy()
        ix = tf.random.categorical(p.reshape(1, -1), num_samples=1)[0,0].numpy()
        return self.itos_[ix]

    def generate(self):
      prev_token = '.'
      results = []
      while True:
          next_token = model(prev_token)
          if next_token == '.':
              break
          results.append(next_token)
          prev_token = next_token
      return ''.join(results)

if __name__ == '__main__':
  words = open('names.txt', 'r').read().splitlines()
  tf.random.set_seed(42)
  model = BigramModel(words)
  for i in range(5):
    print(model.generate())
