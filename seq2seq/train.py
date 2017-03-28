import os
import json
import sys
import config
import tensorflow as tf
import numpy as np
from time import time
from data_helper import DataSet
from model import Model
flags = tf.flags

flags.DEFINE_string("config_name", "", "Configuration name")
FLAGS = flags.FLAGS

if __name__ == '__main__':
    assert FLAGS.config_name
    config = config.configuration[FLAGS.config_name]
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.config_name))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    save_path = os.path.join(checkpoint_dir, "model")
    dev_res_path = os.path.join(out_dir, 'dev.res')
    log_path = os.path.join(out_dir, 'train.log')
    config_path = os.path.join(out_dir, 'config.json')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if config['reload']:
        config['load_path'] = save_path
    else:
        config['load_path'] = None

    dataset = DataSet(config['dictionaries'], config['max_len'], n_words_source=config['source_vocab_size'], n_word_target=config['target_vocab_size'])
    #config['source_vocab_size'] = dataset.num_source_word
    #config['target_vocab_size'] = dataset.num_target_word

    fout_log = open(log_path, 'a')
    with open(config_path, 'w') as fout:
        print >> fout, json.dumps(config)

    model = Model(config)

    for epoch_index in xrange(config['num_epoch']):
        tic = time()
        lno = 0
        total_loss = 0.
        for data in dataset.train_batch_iterator(config['dataset'], config['batch_size']):

            source, target, source_mask, target_mask = data
            if lno % 1000 == 0:
                sys.stdout.write("Process to %d\r" % lno)
                sys.stdout.flush()
            lno += config['batch_size']
            #print model.fit(source, target, source_mask, target_mask)
            loss = model.fit(source, target, source_mask, target_mask)
            total_loss += loss

        info = '# %s: loss = %s, it costs %ss' % (epoch_index, total_loss,  time() - tic)
        print info
        print >> fout_log, info
        old_path = model.save("%s-%s" % (save_path, epoch_index))