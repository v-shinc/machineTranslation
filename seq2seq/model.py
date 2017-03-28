import tensorflow as tf
import numpy as np
import copy


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


def gru_layer(inputs, masks, dim, scope_name):
    # x: [batch_size, seq_len, word_dim] list of [batch_size, word_dim]
    # mask: [batch_size, seq_len]
    seq_len = inputs.get_shape()[1]
    batch_size = tf.shape(inputs)[0]
    inputs = [tf.squeeze(x, [1]) for x in tf.split(1, seq_len, inputs)]
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
                outputs.append(_gru_step(tf.zeros([batch_size, dim]), inputs[t], masks[t]))
            else:
                scope.reuse_variables()
                outputs.append(_gru_step(outputs[-1], inputs[t], masks[t]))
    #hidden_states = tf.scan(_gru_step, [inputs, masks], initializer=tf.zeros([inputs.get_shape()[1], dim]))
    return outputs

def cond_gru_layer(inputs, masks, context, init_state, dim, scope_name, one_step=False):

    def _cond_gru_step(h_prev, x, m, ctx):
        with tf.variable_scope('cond_gru_layer'):

            r, z = tf.split(1, 2, tf.nn.sigmoid(_linear([h_prev, x, ctx], 2 * dim, 'gate')))
            # hidden state proposal
            h = tf.nn.tanh(_linear([x, h_prev * r, ctx], dim, 'state'))
            # leaky integrate and obtain next hidden state
            h = z * h_prev + (1. - z) * h
            if masks != None:
                h = m * h + (1. - m) * h_prev
        return h
    if one_step:
        with tf.variable_scope(scope_name) as scope:
            scope.reuse_variables()
            return [_cond_gru_step(init_state, inputs, None, context)]  # return a tensor
    else:
        seq_len = inputs.get_shape()[1]
        inputs = [tf.squeeze(x, [1]) for x in tf.split(1, seq_len, inputs)]
        masks = tf.split(1, seq_len, masks)
        outputs = []
        with tf.variable_scope(scope_name) as scope:
            for t in xrange(seq_len):
                if t == 0:
                    outputs.append(_cond_gru_step(init_state, inputs[t], masks[t], context))
                else:
                    scope.reuse_variables()
                    outputs.append(_cond_gru_step(outputs[-1], inputs[t], masks[t], context))
        #hidden_states = tf.scan(_cond_gru_step, [inputs, masks], initializer=init_state)
        return outputs  # return list of tensor


class Model:
    def __init__(self, params):
        batch_size = params['batch_size']
        max_len = params['max_len']
        word_dim = params['word_dim']
        hidden_dim = params['hidden_dim']
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
        self.init_state = ff_layer(self.context, params['hidden_dim'], 'ff_state', tf.nn.tanh)

        # word embedding (target)

        y_emb = tf.nn.embedding_lookup(y_embedding, self.y)
        # shift one time step to right
        y_embed_shifted = tf.concat(1, [tf.zeros_like(y_emb[:, 0:1, :]), y_emb[:, :-1, :]]) # [batch_size, seq_len, word_dim]

        # decoder - pass through the decoder gru, recurrence here

        ss_ = cond_gru_layer(y_embed_shifted, self.y_mask, self.context, self.init_state, hidden_dim, 'decoder')  # [seq_len, batch_size, hidden_dim]
        contexts = tf.expand_dims(self.context, 1)  # [batch_size, 1, hidden_dim]
        # compute word probabilities
        ss_ = tf.concat(1, [tf.expand_dims(t, 1) for t in ss_])

        with tf.variable_scope('word_prob'):
            logit_lstm = ff_layer(ss_, word_dim, 'ff_logit_lstm')
            logit_prev = ff_layer(y_embed_shifted, word_dim, 'ff_logit_prev')
            logit_ctx = ff_layer(contexts, word_dim, 'ff_logit_ctx')
            logits = tf.nn.tanh(logit_lstm + logit_prev + logit_ctx)
            logits = ff_layer(logits, params['target_vocab_size'], 'ff_logit')  # [batch_size, seq_len, target_vocab_size]

        # cost
        labels = tf.one_hot(self.y, params['target_vocab_size'])
        probs = tf.reshape(tf.nn.softmax(tf.reshape(logits, [-1, params['target_vocab_size']])), [-1, max_len, params['target_vocab_size']])
        cost = tf.log(tf.reduce_sum(labels * probs, 2))   # [batch_size, seq_len]
        self.cost = - tf.reduce_mean(tf.reduce_sum(cost * self.y_mask, 1))

        # train operator
        self.train_op = tf.train.AdamOptimizer(params['lr']).minimize(self.cost)

        # build graph for beam search
        self.cur_y = tf.placeholder(tf.int32, [None], name='cur_y')
        self.prev_state = tf.placeholder(tf.float32, [None, params['hidden_dim']], name="prev_state")
        self.context_input = tf.placeholder(tf.float32, [None, params['hidden_dim']], name="context")

        self.cur_y_emb = tf.cond(
            (self.cur_y < tf.zeros_like(self.cur_y))[0],
            lambda: tf.zeros([tf.shape(self.cur_y)[0], word_dim]),
            lambda: tf.nn.embedding_lookup(y_embedding, self.cur_y)
        )

        # apply one step of gru layer
        proj = cond_gru_layer(self.cur_y_emb, None, self.context_input, self.prev_state, hidden_dim, "decoder", one_step=True)
        self.next_state = proj[0]
        with tf.variable_scope('word_prob') as scope:
            scope.reuse_variables()

            cur_logit_lstm = ff_layer(self.prev_state, word_dim, "ff_logit_lstm")
            cur_logit_prev = ff_layer(self.cur_y_emb, word_dim, 'ff_logit_prev')
            cur_logit_ctx = ff_layer(self.context_input, word_dim, "ff_logit_ctx")
            cur_logits = tf.nn.tanh(cur_logit_lstm + cur_logit_prev + cur_logit_ctx)
            cur_logits = ff_layer(cur_logits, params['target_vocab_size'], 'ff_logit')  # [batch_size, target_vocab_size]
        self.next_prob = tf.nn.softmax(cur_logits)

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

    def predict(self, x, x_mask, k, max_len):

        sample = []
        sample_score = []
        # beam search
        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype('float32')

        # get initial state of decoder rnn and encoder context

        feed_dict = {self.x: x, self.x_mask: x_mask}
        next_state, context0 = self.session.run([self.init_state, self.context], feed_dict)

        next_word = np.array([-1])
        for t in xrange(max_len):
            ctx = np.tile(context0, [live_k, 1])  # [live_k, hidden_dim]
            next_state, next_prob, cur_y_emb = self.session.run([self.next_state, self.next_prob, self.cur_y_emb], feed_dict={
                self.context_input: ctx,
                self.cur_y: next_word,
                self.prev_state: next_state
            })
            cand_scores = hyp_scores[:, None] - np.log(next_prob)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k - dead_k)]

            voc_size = next_prob.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = np.zeros(k - dead_k).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))

            # check the finished sample
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = np.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break
            next_word = np.array([w[-1] for w in hyp_samples])
            next_state = np.array(hyp_states)
        if len(sample) == 0:
            return hyp_samples, hyp_scores
        return sample, sample_score

    def save(self, save_path):
        return self.saver.save(self.session, save_path)

