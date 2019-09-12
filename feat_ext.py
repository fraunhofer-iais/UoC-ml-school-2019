from pathlib import Path
from itertools import chain, starmap, repeat, zip_longest
from collections import namedtuple
import tensorflow as tf
import numpy as np

from audioset import vggish_params
from audioset import vggish_input
from audioset import vggish_postprocess
from rennet.utils.audio_utils import get_audio_metadata, load_audio
from rennet.datasets.ka3 import ActiveSpeakers, times_for_labelsat
from rennet.utils.np_utils import to_categorical

Feat = namedtuple('Feat', 'Feature_type,fn')

class AudioLabelPair:
    context_features = {
        "filename": Feat(tf.FixedLenFeature([], dtype=tf.string), lambda f, v: f.bytes_list.value.append(v)),
        "length": Feat(tf.FixedLenFeature([], dtype=tf.int64), lambda f, v: f.int64_list.value.append(v)),
        "dims": Feat(tf.FixedLenFeature([], dtype=tf.int64), lambda f, v: f.int64_list.value.append(v)),
    }

    sequence_features = {
        "embedding": Feat(tf.FixedLenSequenceFeature([128], dtype=tf.float32), lambda f, v: f.float_list.value.extend(v)),
        "speechact": Feat(tf.FixedLenSequenceFeature([2], dtype=tf.float32), lambda f, v: f.float_list.value.extend(v)),
    }

    def __init__(self, path_to_audio, path_to_label, labels_parser=ActiveSpeakers.from_file):
        if isinstance(path_to_audio, Path):
            path_to_audio = str(path_to_audio.absolute())
        if isinstance(path_to_label, Path):
            path_to_label = str(path_to_label.absolute())

        self.audio = get_audio_metadata(path_to_audio)
        self.label = labels_parser(path_to_label)

    def __repr__(self):
        return f'AudioLabelFilePair: [\naudio:\n{self.audio},\n\nlabel:\n{self.label[:min(5, len(self.label))]} ...\n]'

    def to_vggish_SequenceExample(self, sess, post_processor):
        features = self._get_audio_examples()

        embedding = examples_to_embedding(features, sess, post_processor)

        labels = self._get_label_examples()

        empty_embedding = post_processor.postprocess(np.zeros(embedding.shape[1:])[None, ...])[0, ...]
        empty_label = np.zeros(labels.shape[1:])

        return self._to_SequenceExample(embedding, labels, empty_embedding, empty_label)

    def _get_audio_examples(self):
        s, e = self._get_start_end_seconds()
        return audiofile_to_examples(self.audio.filepath, start=s, end=e)

    def _get_label_examples(self):
        s, e = self._get_start_end_seconds()
        with self.label.samplerate_as(1.0):
            timestamps = times_for_labelsat(
                (e - s), 
                vggish_params.SAMPLE_RATE, 
                vggish_params.EXAMPLE_HOP_SECONDS, 
                vggish_params.EXAMPLE_WINDOW_SECONDS
            )
            with self.label.min_start_as(0.0):
                labels = self.label.labels_at(timestamps)
        
        return to_categorical(labels.sum(axis=1).clip(0, 1), nclasses=2)

    def _get_start_end_seconds(self):
        with self.label.samplerate_as(1.0):  # in seconds
            s = max(0.0, self.label.min_start)  # first timestamp for which there's a label
            e = min(self.audio.nsamples / self.audio.samplerate, self.label.max_end)  # last end-time for which there's a label

        return s, e

    def _to_SequenceExample(self, embedding, labels, empty_embedding, empty_label):
        ex = tf.train.SequenceExample()
        AudioLabelPair.context_features['filename'].fn(
            ex.context.feature['filename'], 
            bytes(Path(self.audio.filepath).name.split(".")[0], 'utf')
        )
        AudioLabelPair.context_features['length'].fn(
            ex.context.feature['length'], 
            max(len(embedding), len(labels))
        )
        AudioLabelPair.context_features['dims'].fn(
            ex.context.feature['dims'], 
            embedding.shape[1]
        )

        embedding_flist = ex.feature_lists.feature_list['embedding']
        speechact_flist = ex.feature_lists.feature_list['speechact']

        embedding_fn = AudioLabelPair.sequence_features['embedding'].fn
        speechact_fn = AudioLabelPair.sequence_features['speechact'].fn

        for (e, l) in zip_longest(embedding, labels):
            embedding_fn(embedding_flist.feature.add(), e if e is not None else empty_embedding)
            speechact_fn(speechact_flist.feature.add(), l if l is not None else empty_label)

        return ex

    @classmethod
    def all_in_dir(cls, 
                   path_to_dir, 
                   audio_filename_pattern, 
                   label_filename_pattern, 
                   labels_parser=ActiveSpeakers.from_file):
        
        if not isinstance(path_to_dir, Path):
            path_to_dir = Path(path_to_dir)

        if isinstance(audio_filename_pattern, str):
            audio_filename_pattern = [audio_filename_pattern]

        if isinstance(label_filename_pattern, str):
            label_filename_pattern = [label_filename_pattern]

        assert path_to_dir.is_dir(), f'path_to_dir at: {path_to_dir} is not a valid directory'

        audio_files = sorted(chain(*map(path_to_dir.glob, audio_filename_pattern)))
        label_files = sorted(chain(*map(path_to_dir.glob, label_filename_pattern)))

        assert len(audio_files) == len(label_files),\
            f"Mismatch in number of audio files found vs. label files: {len(audio_files)} vs. {len(label_files)}"

        # check if each audio has a label file, based on filename alone
        assert all(
            af.name.split('.')[0] == lf.name.split('.')[0] and af.is_file() and lf.is_file()
            for (af, lf) in zip(audio_files, label_files)
        ), 'Mimatch in pairings of audio files and label files. Not all pairs present (based on filename without extension)'

        return list(starmap(AudioLabelPair, zip(audio_files, label_files, repeat(labels_parser))))


def audiofile_to_examples(filepath, start=0.0, end=None):  # end=None by default reads the entire audio to example
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    samples = load_audio(
        str(filepath.absolute()), 
        samplerate=vggish_params.SAMPLE_RATE, 
        mono=True, 
        offset=start, 
        duration=(end - start) if end is not None else None,
    )
    return vggish_input.waveform_to_examples(samples, vggish_params.SAMPLE_RATE)


def get_post_processor(filepath):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    return vggish_postprocess.Postprocessor(str(filepath.absolute()))


def examples_to_embedding(features, sess, post_processor):
    features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

    [raw_embeddings] = sess.run([embedding_tensor], feed_dict={features_tensor: features})
    embedding = post_processor.postprocess(raw_embeddings)

    return embedding