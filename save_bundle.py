from __future__ import print_function
import tensorflow as tf

import argparse
import os
from six.moves import cPickle

from model import Model


def main():
    parser = argparse.ArgumentParser(description="Save a trained model as a bundle.")
    parser.add_argument('--out_dir', type=str, default="bundle",
                        help='folder to which to save the bundle')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory from which to load checkpointed models')

    args = parser.parse_args()
    save_bundle(args)


def save_bundle(args):
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, _ = cPickle.load(f)

    model = Model(saved_args, training=False)

    # write chars
    with open(os.path.join(args.out_dir, "chars.bin"), "wb") as df:
        for c in chars:
            df.write(c.encode('utf-8'))

    builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(args.out_dir, "model"))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(
            model.input_data)

        predict_initial_state_tensor_info = tf.saved_model.utils.build_tensor_info(
            model.state_placeholder)

        predict_final_state_tensor_info = tf.saved_model.utils.build_tensor_info(
            tf.identity(model.final_state, "final_state"))

        predict_probability_tensor_info = tf.saved_model.utils.build_tensor_info(
            tf.identity(model.probs, "probs"))

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    tf.saved_model.signature_constants.PREDICT_INPUTS:
                        predict_inputs_tensor_info,
                    "initial_state": predict_initial_state_tensor_info,
                },
                outputs={
                    tf.saved_model.signature_constants.PREDICT_OUTPUTS:
                        predict_probability_tensor_info,
                    "final_state": predict_final_state_tensor_info,
                },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        # Save the variables to disk.
        builder.add_meta_graph_and_variables(
            sess, ["char-level"],
            signature_def_map={
                'predict_next_char': prediction_signature,
            })
        save_path = builder.save()

        print("Model saved in file: %s" % save_path)

if __name__ == '__main__':
    main()
