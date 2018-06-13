def read_data(file_name, target_case="term_of_imprisonment"):
    import json
    contents, labels = [], []
    print('getting data from'+file_name)
    with open(file=file_name, mode='r', encoding='utf8') as f:
        for line in f:
            content = str(json.loads(line)["fact"]).replace(
                '，', '').replace(' ', '')
            label = json.loads(line)["meta"][target_case]
            if target_case == "term_of_imprisonment":
                if label["death_penalty"]:
                    label = 301
                elif label["life_imprisonment"]:
                    label = 302
                else:
                    label = label["imprisonment"]
            else:
                temp = str(label[0]).replace(
                    '[', '').replace('，', '').replace(' ', '')
                label = temp.replace(']', '')
            if content:
                contents.append(content)
                labels.append(label)
    return contents, labels


def build_vocab(train_dir, valid_dir, test_dir, vocab_dir, vocab_size, min_frequence, split=False):
    from collections import Counter
    import jieba
    contents, _ = read_data(valid_dir)
    temp, _ = read_data(test_dir)
    contents += temp
    temp, _ = read_data(train_dir)
    contents += temp
    all_data = []
    if split:
        temp_for_split = str()
        for index, line in enumerate(contents):
            temp_for_split = temp_for_split+'\n'+line
            if index % 10000 == 0:
                print(index/10000)
            if index & 30000 == 0:
                split_content = " ".join(jieba.cut(temp_for_split))
                with open(file='temp', mode='a', encoding='utf8') as temp_file:
                    temp_file.write(split_content)
                temp_for_split = ''
        with open(file='temp', mode='r', encoding='utf8') as temp_file:
            for line in temp_file.readlines():
                all_data.extend(line.strip().split(' '))
    else:
        for content in contents:
            all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size-1)
    vocab_list = list(zip(*count_pairs))
    if min_frequence >= 0:
        index = vocab_list[1].index(min_frequence)
    else:
        index = len(vocab_list[0])
    words = vocab_list[0]
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)[:index]
    with open(vocab_dir, mode='w', encoding='utf8') as f:
        f.write('\n'.join(words) + '\n')
    print('build success')


def read_vocab(vocab_dir):
    with open(vocab_dir, mode='r', encoding='utf8') as fp:
        words = [_.strip() for _ in fp.readlines()]
        word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_word2vec(vocab_dir):
    with open(vocab_dir, mode='r', encoding='utf8') as fp:
        vocab_length, vocab_dim = 0, 0
        words, vectors = [], []
        for line in fp.readlines():
            if vocab_length == 0:
                vocab_length, vocab_dim = line.strip().split(' ')
            else:
                words.append(line.strip().split(' ')[0])
                vectors.append(line.strip().split(' ')[1:])
        word_to_id = dict(zip(words, vectors))
    return words, word_to_id


def batch_iter(x, y, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    import numpy as np
    data_size = len(x)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        x_shuffle = x[shuffle_indices]
        y_shuffle = y[shuffle_indices]
    else:
        x_shuffle = x
        y_shuffle = y
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield x_shuffle[start_index:end_index], y_shuffle[start_index:end_index]


def get_data_with_vocab(data_dir, words_to_id, cat_to_id, config, target_case='term_of_imprisonment'):
    import keras
    import jieba
    contents, labels = read_data(data_dir, target_case)
    data_id, label_id = [], []
    for i in range(len(contents)):
        sentence_id = []
        if config.hierachy_init:
            sentences = contents[i].split('。')
            for _, sentence in enumerate(sentences):
                sentence_id.append([words_to_id[x]
                                    for x in sentence if x in words_to_id])
            sentence_pad = keras.preprocessing.sequence.pad_sequences(
                sentence_id, int(config.seq_length), dtype='float32')
            data_id.append(sentence_pad)
        else:
            data_id.append([words_to_id[x]
            for x in jieba.cut(contents[i]) if x in words_to_id])
        if target_case != 'term_of_imprisonment':
            label_id.append(cat_to_id[labels[i]])
        else:
            label_id.append(labels[i])

    x_data = keras.preprocessing.sequence.pad_sequences(
        data_id, int(config.para_length), dtype='float32')
    y_data = keras.utils.to_categorical(label_id, num_classes=303)
    return x_data, y_data

def split_data(train_dir, val_dir, test_dir):
    import jieba
    import json
    contents=[]
    with open(file=val_dir+'.json',mode='r') as input_file:
        for line in input_file.readlines():
            temp=jieba.cut(str(json.loads(line)["fact"]).replace(
                '，', '').replace(' ', ''))
            meta = json.loads(line)["meta"]
            contents.append(json.dumps({"fact":temp,"meta":meta}))
    with open(file=val_dir+'_split.json',mode='w') as output_file:
        for _, content in enumerate(contents):
            output_file.write(content+'/n')

def to_words(content, words):
    return ' '.join(words[x] for x in content)


def read_catagory(cat_dir):
    cat_to_id = {}
    catagories = []
    with open(file=cat_dir, mode='r', encoding='utf8') as cat_file:
        index = 1
        for line in cat_file.readlines():
            catagory = line.strip()
            catagories.append(catagory)
            cat_to_id[catagory] = index
            index += 1
    return catagories, cat_to_id


def balance_data(base_dir):
    import random
    file = 'felony'
    file_name = base_dir+file+'_train.txt'
    contents, labels = read_data(file_name)
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    all_data, balance_data = [], []
    for i, label in enumerate(labels):
        count[int(label)-10] += 1
        all_data.append(contents[i]+','+labels[i])
    balance = min(count)
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    random.shuffle(all_data)
    random.shuffle(all_data)
    random.shuffle(all_data)
    for _, label in enumerate(all_data):
        content, number = label.split(',')
        number = int(number)
        if count[number-10] < balance and len(content) > 30:
            count[number-10] += 1
            balance_data.append(label)
    with open(file=base_dir+file+'_balance.txt', mode='w', encoding='utf8') as f:
        for i, label in enumerate(balance_data):
            f.write(label+'\n')


def main():
    build_vocab('./good/data_train.json', './good/data_valid.json', './good/data_test.json',
                './good/vocab.txt', vocab_size=7000, min_frequence=-1, split=False)


if __name__ == '__main__':
    main()
