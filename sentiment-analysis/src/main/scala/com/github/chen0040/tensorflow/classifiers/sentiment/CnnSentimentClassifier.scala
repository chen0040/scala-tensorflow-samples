package com.github.chen0040.tensorflow.classifiers.sentiment

import org.tensorflow.{Session, Tensor}

class CnnSentimentClassifier extends SentimentClassifier {
  override def runPredict(textTensor: Tensor[java.lang.Float], sess: Session): Tensor[java.lang.Float] = {
    sess.runner.feed("embedding_1_input:0", textTensor).feed("spatial_dropout1d_1/keras_learning_phase:0", Tensor.create(false)).fetch("output_node0:0").run.get(0).expect(classOf[java.lang.Float])
  }
}
