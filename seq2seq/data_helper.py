import numpy as np
def build_dictionary(fn_word_count):
    index = 2
    word_to_id = dict()
    id_to_word = dict()
    with open(fn_word_count) as fin:
        for line in fin:
            w, cnt = line.decode('utf8').strip().split()
            word_to_id[w] = index
            id_to_word[index] = w
            index += 1
    word_to_id['<eos>'] = 0
    word_to_id['<unk>'] = 1
    id_to_word[0] = '<eos>'
    id_to_word[1] = '<unk>'
    return word_to_id, id_to_word


def gen_word_count_list(fn_list, fn_out):
    word_count = dict()
    for fn in fn_list:
        with open(fn) as fin:
            for line in fin:
                ws = line.strip().split()
                for w in ws:
                    if w not in word_count:
                        word_count[w] = 1
                    else:
                        word_count[w] += 1

    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    with open(fn_out, 'w') as fout:
        for w, cnt in word_count:
            print >> fout, w, cnt


class LineIterator:
    def __init__(self, fn_list):
        self.files = [open(fn) for fn in fn_list]

        self.cur_index = 0

    def __iter__(self):
        return self

    def reset(self):
        for i in range(len(self.files)):
            self.files[i].seek(0)
        self.cur_index = 0

    def next(self):
        if self.cur_index == len(self.files):
            self.reset()
            raise StopIteration
        line = self.files[self.cur_index].readline()
        if line == '':
            self.cur_index += 1
            return self.next()
        return line.strip()


class DataSet:
    def __init__(self, dictionaries, max_len, n_words_source=-1, n_word_target=-1):

        self.source_word2id, self.source_id2word = build_dictionary(dictionaries[0])
        self.target_word2id, self.target_id2word = build_dictionary(dictionaries[1])
        self.max_len = max_len
        self.n_word_source = n_words_source
        self.n_word_target = n_word_target

    @property
    def num_source_word(self):
        return self.n_word_source if self.n_word_source >= 0 else len(self.source_word2id)
    @property
    def num_target_word(self):
        return self.n_word_target if self.n_word_target >= 0 else len(self.target_word2id)

    def train_batch_iterator(self, dataset, batch_size):
        sources = LineIterator([fn for fn, _ in dataset])
        targets = LineIterator([fn for _, fn in dataset])
        source_buff = []
        target_buff = []
        source_mask = []
        target_mask = []
        for s in sources:
            t = targets.next().split()

            ss = []
            for w in s.split():

                if w in self.source_word2id:
                    ss.append(self.source_word2id[w])
                else:
                    ss.append(1)
            if len(ss) > self.max_len:
                continue
            tt = []
            for w in t:
                if w in self.target_word2id:
                    tt.append(self.target_word2id[w])
                else:
                    tt.append(1)
            if len(tt) > self.max_len:
                continue
            # pad batches
            source_mask.append([1] * len(ss) + [0] * (self.max_len - len(ss)))
            target_mask.append([1] * len(tt) + [0] * (self.max_len - len(tt)))
            ss += (self.max_len - len(ss)) * [0]
            tt += (self.max_len - len(tt)) * [0]

            if self.n_word_source >= 0:
                ss = [i if i < self.n_word_source else 1 for i in ss]
            if self.n_word_target >= 0:
                tt = [i if i < self.n_word_target else 1 for i in tt]
            source_buff.append(ss)
            target_buff.append(tt)
            if len(source_buff) == batch_size:
                yield [np.array(e) for e in source_buff, target_buff, source_mask, target_mask]
                source_buff = []
                target_buff = []
                source_mask, target_mask = [], []



if __name__ == '__main__':
    # gen_word_count_list(['../data/europarl-v7.fr-en.en.tok.shuf', "../data/newstest2011.en.tok"], "../data/word.count.en")
    # gen_word_count_list(['../data/europarl-v7.fr-en.fr.tok.shuf', "../data/newstest2011.fr.tok"], '../data/word.count.fr')
    # line_iter = LineIterator(['tmp'])
    # count = 0
    # for epoch in range(3):
    #     for line in line_iter:
    #         print line

    dataset = DataSet(("../data/word.count.en", '../data/word.count.fr'), 100)
    i = 0
    for batch in dataset.train_batch_iterator([('../data/europarl-v7.fr-en.en.tok.shuf', '../data/europarl-v7.fr-en.fr.tok.shuf')], 20):
        x, y, x_mask, y_mask = batch
        print x.shape
        print i
        i+= 1