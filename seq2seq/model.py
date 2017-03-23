import tensorflow as tf



def _linear(args, output_size, scope):
    # list of tensors with shape [batch_size, input_dim]
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)

    total_arg_size = sum([shape[1].value for shape in shapes])
    dtype = args[0].dtype

    with tf.variable_scope(scope):
        weights = tf.get_variable('weights', [total_arg_size, output_size], dtype=dtype)
        res = tf.matmul(tf.concat(1, args), weights)   # old API
        biases = tf.get_variable('biases', [output_size], dtype=dtype,
                                 initializer=tf.constant_initializer(0, dtype=dtype))
        return tf.add(res, biases)

def ff_layer(inputs, dim, scope, activation=None):
    shape = inputs.get_shape()
    shape = [-1 if not s.value else s.value for s in shape]
    if len(shape) > 2:
        inputs = tf.reshape(inputs, [-1, shape[-1]])
    with tf.variable_scope(scope):
        weight = tf.get_variable('weight', [shape[-1], dim], dtype=inputs.dtype)
        bias = tf.get_variable('bias', [dim], dtype=inputs.dtype)
        res = tf.add(tf.matmul(inputs, weight), bias)
    if activation:
        res = activation(res)
    if len(shape) > 2:
        shape[-1] = dim
        res = tf.reshape(res, shape)
    return res

def _shift(tensor):
    # tensor: [batch_size, seq_len, dim]
    shape = [s.value for s in tensor.get_shape()]
    shape[1] = 1
    res = tf.concat(1, [tf.zeros(shape),  tensor[:, -1]])
    return res

def gru_layer(inputs, masks, dim, scope_name):
    # x: [batch_size, seq_len, word_dim] list of [batch_size, word_dim]
    # mask: [batch_size, seq_len]
    seq_len = inputs.get_shape()[1]
    inputs = [tf.squeeze(x) for x in tf.split(1, seq_len, inputs)]
    masks = tf.split(1, seq_len, masks) # list of [batch_size, 1]
    def _gru_step(h_prev, x, m):
        # reset and update gates
        with tf.variable_scope('gru_step'):
            r, z = tf.split(1, 2, tf.nn.sigmoid(_linear([h_prev, x], 2 * dim, 'gate')))
            # hidden state proposal

            h = tf.nn.tanh(_linear([x, h_prev * r], dim, 'state'))
            # leaky integrate and obtain next hidden state
            h = z * h_prev + (1. - z) * h  # [batch_size, dim]
            h = m * h + (1. - m) * h_prev
        return h
    outputs = []
    with tf.variable_scope(scope_name) as scope:
        for t in xrange(seq_len):
            if t == 0:
                outputs.append(_gru_step(tf.zeros([inputs[0], dim]), inputs[t], masks[t]))
            else:
                scope.reuse_variables()
                outputs.append(_gru_step(outputs[-1], inputs[t], masks[t]))
    #hidden_states = tf.scan(_gru_step, [inputs, masks], initializer=tf.zeros([inputs.get_shape()[1], dim]))
    return outputs

def cond_gru_layer(inputs, masks, context, init_state, dim, scope_name):
    seq_len = inputs.get_shape()[1]
    inputs = [tf.squeeze(x) for x in tf.split(1, seq_len, inputs)]
    masks = tf.split(1, seq_len, masks)
    def _cond_gru_step(h_prev, x, m):
        with tf.variable_scope('cond_gru_layer'):

            r, z = tf.split(1, 2, tf.nn.sigmoid(_linear([h_prev, x, context], 2 * dim, 'gate')))
            # hidden state proposal
            h = tf.nn.tanh(_linear([x, h_prev * r, context], dim, 'state'))
            # leaky integrate and obtain next hidden state
            h = z * h_prev + (1. - z) * h
            h = m * h + (1. - m) * h_prev
        return h
    outputs = []
    with tf.variable_scope(scope_name) as scope:
        for t in xrange(seq_len):
            if t == 0:
                outputs.append(_cond_gru_step(init_state, inputs[t], masks[t]))
            else:
                scope.reuse_variables()
                outputs.append(_cond_gru_step(outputs[-1], inputs[t], masks[t]))
    #hidden_states = tf.scan(_cond_gru_step, [inputs, masks], initializer=init_state)
    return outputs


class Model:
    def __init__(self, params):
        batch_size = params['batch_size']
        max_len = params['max_len']
        self.x = tf.placeholder(tf.int32, [None, max_len], name='x')
        self.x_mask = tf.placeholder(tf.float32, [None, max_len], name='x_mask')
        self.y = tf.placeholder(tf.int32, [None, max_len], name='y')
        self.y_mask = tf.placeholder(tf.float32, [None, max_len], name='y_mask')

        with tf.variable_scope('word_embedding'):
            initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
            x_embedding = tf.get_variable('x_embedding', [params['source_vocab_size'], params['word_dim']], initializer=initializer)
            y_embedding = tf.get_variable('y_embedding', [params['target_vocab_size'], params['word_dim']], initializer=initializer)
        # word embedding (source)
        x_emb = tf.nn.embedding_lookup(x_embedding, self.x)
        # pass through encoder gru, recurrence here
        hh = gru_layer(x_emb, self.x_mask, params['hidden_dim'], 'encoder')  # [seq_len, batch_size, hidden_dim]
        # last hidden state of encoder rnn will be used to initialize decoder rnn

        self.context = hh[-1] #[batch_size, hidden_dim]
        # initial decoder state
        init_state = ff_layer(self.context, params['hidden_dim'], 'ff_state', tf.nn.tanh)

        # word embedding (target)
        self.y_emb = tf.nn.embedding_lookup(y_embedding, self.y)
        self.y_embed_shifted = _shift(self.y_emb) # [batch_size, seq_len, word_dim]

        # decoder - pass through the decoder gru, recurrence here

        self.ss_ = cond_gru_layer(self.y_embed_shifted, self.y_mask, self.context, init_state, params['hidden_dim'], 'decoder')  # [seq_len, batch_size, hidden_dim]
        contexts = tf.expand_dims(self.context, 1) # [batch_size, 1, hidden_dim]
        # compute word probabilities
        logit_lstm = ff_layer(tf.transpose(tf.pack(self.ss_), [0, 1]), params['word_dim'], 'ff_logit_lstm')
        logit_prev = ff_layer(self.y_embed_shifted, [0, 1], params['word_dim'], 'ff_logit_prev')
        logit_ctx = ff_layer(contexts, params['word_dim'], 'ff_logit_ctx')
        logits = tf.nn.tanh(logit_lstm + logit_prev + logit_ctx)
        logits = ff_layer(logits, params['target_vocab_size'], 'ff_logit')  # [batch_size, seq_len, target_vocab_size]

        # cost
        labels = tf.one_hot(self.y, params['target_vocab_size'])
        probs = tf.reshape(tf.nn.softmax(tf.reshape(logits, [-1, params['target_vocab_size']])), [-1, max_len, params['target_vocab_size']])
        cost = tf.log(tf.reduce_sum(labels * probs, 2)) # [batch_size, seq_len]
        self.cost = - tf.reduce_mean(tf.reduce_sum(cost * self.y_mask, 1))

        # train operator
        self.train_op = tf.train.GradientDescentOptimizer(params['lr']).minimize(self.cost)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.session = tf.Session(config=config)
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
        if params['load_path']:
            self.saver.restore(self.session, params['load_path'])
        else:
            self.session.run(tf.initialize_all_variables())

    def fit(self, x, y, x_mask, y_mask):
        feed_dict = dict()
        feed_dict[self.x] = x
        feed_dict[self.y] = y
        feed_dict[self.x_mask] = x_mask
        feed_dict[self.y_mask] = y_mask
        _, loss = self.session.run([self.train_op, self.cost], feed_dict)
        return loss

    def predict(self):
        pass

    def save(self, save_path):
        return self.saver.save(self.session, save_path)

