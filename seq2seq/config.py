configuration = dict()

configuration['seq2seq'] = {
    "gpu": 1,
    'dataset': [('../data/europarl-v7.fr-en.en.tok.shuf', '../data/europarl-v7.fr-en.fr.tok.shuf')],
    'valid_dataset': [('../data/newstest2011.en.tok', '../data/newstest2011.fr.tok')],
    'dictionaries': ('../data/word.count.en', '../data/word.count.fr'),
    'max_len': 25,
    'batch_size': 30,
    'num_epoch': 100,
    'lr': 0.001,
    'dropout_keep_rate': 0.5,
    'word_dim': 32,
    'hidden_dim': 32,
    'reload': False
}