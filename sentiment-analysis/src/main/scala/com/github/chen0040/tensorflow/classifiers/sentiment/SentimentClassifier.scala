package com.github.chen0040.tensorflow.classifiers.sentiment

import org.tensorflow.{Graph, Session, Tensor}

import java.io.IOException
import java.io.InputStream

import com.github.chen0040.tensorflow.classifiers.utils.{InputStreamUtils, ResourceUtils, TextModel}
import org.slf4j.LoggerFactory

object SentimentClassifier {
  private val logger = LoggerFactory.getLogger(classOf[SentimentClassifier])
}

abstract class SentimentClassifier extends AutoCloseable {
  private var graph = new Graph
  private var textModel: TextModel = _

  @throws[IOException]
  def load_model(inputStream: InputStream): Unit = {
    val bytes = InputStreamUtils.getBytes(inputStream)
    graph.importGraphDef(bytes)
  }

  def load_vocab(inputStream: InputStream): Unit = {
    textModel = ResourceUtils.getTextModel(inputStream)
  }

  def predict_label(text: String): String = {
    val predicted = predict(text)
    var argmax = 0
    var max = predicted(0)
    for(i <- 1 until predicted.length) {
      if(predicted(i) > max) {
        max = predicted(i)
        argmax = i
      }
    }
    textModel.toLabel(argmax)
  }

  def predict(text: String): Array[Float] = {

    val textTensor: Tensor[java.lang.Float] = textModel.toTensor(text)
    try {
      val sess = new Session(graph)
      val result = runPredict(textTensor, sess)
      try {
        val rshape = result.shape
        if (result.numDimensions != 2 || rshape(0) != 1) throw new RuntimeException(String.format("Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s", java.util.Arrays.toString(rshape)))
        val nlabels: Int = rshape(1).toInt
        val predicted = result.copyTo(Array.ofDim[Float](1, nlabels))(0)
        return predicted
      } catch {
        case ex: Exception =>
          SentimentClassifier.logger.error("Failed to predict image", ex)
      } finally {
        if (sess != null) sess.close()
        if (result != null) result.close()
      }
    }
    Array.ofDim[Float](2)
  }

  def runPredict(textTensor: Tensor[java.lang.Float], sess: Session): Tensor[java.lang.Float]

  @throws[Exception]
  override def close(): Unit = if (graph != null) {
    graph.close()
    graph = null
  }
}