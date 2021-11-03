import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
import sklearn.metrics

class PassageREDataset(data.Dataset):
    """
    Bag-level relation extraction dataset. Note that relation of NA should be named as 'NA'.
    """

    def __init__(self, path, rel2id, tokenizer):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        super().__init__()
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.num_classes = len(rel2id)
        #self.bag_size = bag_size
        self.id2rel = {}
        for k,v in self.rel2id.items():
            self.id2rel[v] = k
        
        # Load the file
        f = open(path)
        self.data = []
        for line in f:
            line = line.rstrip()

            if len(line) > 0:
                self.data.append(eval(line))

        f.close()

        # Construct bag-level dataset (a bag contains instances sharing the same entity-pair)

        self.weight = np.ones((len(self.rel2id)), dtype=np.float32)
        self.bag_scope = []
        self.rel_scope = []
        self.name2id = {}
        self.bag_name = []
        self.facts = {}
        for idx, item in enumerate(self.data):
            fact = (item['h']['id'], item['t']['id'],item['relation']) 
            if item['relation'] != 'NA':
                self.facts[fact] = 1
            name = (item['h']['id'], item['t']['id'])
            if name not in self.name2id:
                self.name2id[name] = len(self.name2id)
                self.bag_scope.append([])
                self.rel_scope.append(set())
                self.bag_name.append(name)
            self.bag_scope[self.name2id[name]].append(idx)
            rel_id = self.rel2id[item['relation']]
            if rel_id not in self.rel_scope[self.name2id[name]]:
                self.rel_scope[self.name2id[name]].add(rel_id)
                self.weight[rel_id] += 1.0
        self.weight = np.float32(1.0 / (self.weight ** 0.05))
        self.weight = torch.from_numpy(self.weight)

    def __len__(self):
        return len(self.bag_scope)

    def __getitem__(self, index):
        bag = self.bag_scope[index]
        random.shuffle(bag)
        rel = torch.LongTensor(list(self.rel_scope[index]))
        onehot_rel = torch.zeros(self.num_classes)
        onehot_rel = onehot_rel.scatter_(0, rel, 1)

        token, mask = self.tokenizer(bag, self.data)
        seqs = [[], []]
        seqs[0].append(token)
        seqs[1].append(mask)
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0)  # (n, L), n is the size of bag
        return [onehot_rel, self.bag_name[index]] + seqs


    def collate_bag_size_fn(data):
        data = list(zip(*data))
        label, bag_name = data[:2]
        seqs = data[2:]
        for i in range(len(seqs)):
            seqs[i] = torch.stack(seqs[i], 0)  # (batch, bag, L)
        return [label, bag_name] + seqs

    def eval(self, pred_result, threshold=0.5):
        """
        Args:
            pred_result: a list with dict {'entpair': (head_id, tail_id), 'relation': rel, 'score': score}.
                Note that relation of NA should be excluded.
        Return:
            {'prec': narray[...], 'rec': narray[...], 'mean_prec': xx, 'f1': xx, 'auc': xx}
                prec (precision) and rec (recall) are in micro style.
                prec (precision) and rec (recall) are sorted in the decreasing order of the score.
                f1 is the max f1 score of those precison-recall points
        """
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec = []
        rec = []
        correct = 0
        total = len(self.facts)

        entpair = {}

        for i, item in enumerate(sorted_pred_result):
            # Save entpair label and result for later calculating F1
            idtf = item['entpair'][0] + '#' + item['entpair'][1]
            if idtf not in entpair:
                entpair[idtf] = {
                    'label': np.zeros((len(self.rel2id)), dtype=np.int),
                    'pred': np.zeros((len(self.rel2id)), dtype=np.int),
                    'score': np.zeros((len(self.rel2id)), dtype=np.float)
                }
            if (item['entpair'][0], item['entpair'][1], item['relation']) in self.facts:
                correct += 1
                entpair[idtf]['label'][self.rel2id[item['relation']]] = 1
            if item['score'] >= threshold:
                entpair[idtf]['pred'][self.rel2id[item['relation']]] = 1
            entpair[idtf]['score'][self.rel2id[item['relation']]] = item['score']

            prec.append(float(correct) / float(i + 1))
            rec.append(float(correct) / float(total))
            
        auc = sklearn.metrics.auc(x=rec, y=prec)
        np_prec = np.array(prec)
        np_rec = np.array(rec) 
        max_micro_f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        best_threshold = sorted_pred_result[(2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).argmax()]['score']
        mean_prec = np_prec.mean()

        label_vec = []
        pred_result_vec = []
        score_vec = []
        for ep in entpair:
            label_vec.append(entpair[ep]['label'])
            pred_result_vec.append(entpair[ep]['pred'])
            score_vec.append(entpair[ep]['score'])
        label_vec = np.stack(label_vec, 0)
        pred_result_vec = np.stack(pred_result_vec, 0)
        score_vec = np.stack(score_vec, 0)

        micro_p = sklearn.metrics.precision_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='micro')
        micro_r = sklearn.metrics.recall_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='micro')
        micro_f1 = sklearn.metrics.f1_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='micro')

        macro_p = sklearn.metrics.precision_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='macro')
        macro_r = sklearn.metrics.recall_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='macro')
        macro_f1 = sklearn.metrics.f1_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='macro')
        
        pred_result_vec = score_vec >= best_threshold
        max_macro_f1 = sklearn.metrics.f1_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='macro')
        max_micro_f1_each_relation = {}
        for rel in self.rel2id:
            if rel != 'NA':
                max_micro_f1_each_relation[rel] = sklearn.metrics.f1_score(label_vec, pred_result_vec, labels=[self.rel2id[rel]], average='micro')

        return {'np_prec': np_prec, 'np_rec': np_rec, 'max_micro_f1': max_micro_f1, 'max_macro_f1': max_macro_f1, 'auc': auc, 'p@100': np_prec[99], 'p@200': np_prec[199], 'p@300': np_prec[299], 'avg_p300': (np_prec[99] + np_prec[199] + np_prec[299]) / 3, 'micro_f1': micro_f1, 'macro_f1': macro_f1, 'max_micro_f1_each_relation': max_micro_f1_each_relation}


def PassageRELoader(path, rel2id, tokenizer, batch_size, shuffle, num_workers=1):
    collate_fn = PassageREDataset.collate_bag_size_fn
    dataset = PassageREDataset(path, rel2id, tokenizer)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn = collate_fn)
    return data_loader
