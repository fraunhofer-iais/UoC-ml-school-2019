import tensorflow as tf

import feat_ext as fx


def parse(proto):
    context_features = {k: v.Feature_type for k, v in fx.AudioLabelPair.context_features.items()}
    
    sequence_features = {k: v.Feature_type for k, v in fx.AudioLabelPair.sequence_features.items()}
    
    context, sequence = tf.parse_single_sequence_example(
        proto,
        context_features=context_features,
        sequence_features=sequence_features,
    )
    
    return context, sequence

def parse_embedding_labels(proto):
    _, sequence = parse(proto)
    return sequence['embedding'], sequence['speechact']

def parse_length(proto):
    context, _ = parse(proto)
    return context['length']

def dataset_shape(fp_split):
    d = tf.data.TFRecordDataset(str(fp_split.absolute()), num_parallel_reads=4)
    d = d.map(parse_length)
    n = d.make_one_shot_iterator().get_next()
    
    num = 0
    with tf.Session() as sess:
        while True:
            try:
                num += sess.run(n)
            except tf.errors.OutOfRangeError:
                break
            except tf.errors.DataLossError:
                print("WARNING: DataLossError encountered. Would not read further.")
                break
                
    return num

def get_dataset(fp_split, batchsize=128):
    dataset = tf.data.TFRecordDataset(str(fp_split.absolute()))
    dataset = dataset.map(parse_embedding_labels)
    dataset = dataset.apply(tf.data.experimental.unbatch())
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(buffer_size=batchsize*8)
    
    return dataset