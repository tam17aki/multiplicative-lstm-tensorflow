# multiplicative-lstm-tensorflow
An implementation of multiplicative LSTM in TensorFlow

#### Example usage:
```
import tensorflow as tf
from MultiplicativeLSTMCell import MultiplicativeLSTMCell

with tf.variable_scope("mlstm) as scope:
    lstm_size = 256
    lstm_init_value = tf.placeholder(tf.float32, shape=(None, 2*lstm_size))
    lstm_cell = MultiplicativeLSTMCell(self.lstm_size, forget_bias=1.0, state_is_tuple=False)
    lstm_outputs, lstm_new_state = tf.nn.dynamic_rnn(lstm_cell,xinput,dtype=tf.float32,initial_state=lstm_init_value)
```
