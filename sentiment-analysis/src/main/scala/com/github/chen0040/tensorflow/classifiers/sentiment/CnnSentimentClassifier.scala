package com.github.chen0040.tensorflow.classifiers.sentiment

import org.tensorflow.{Graph, Session, Tensor}

import java.io.IOException
import java.io.InputStream

import com.github.chen0040.tensorflow.classifiers.utils.{InputStreamUtils, ResourceUtils, TextModel}
import org.slf4j.LoggerFactory

object CnnSentimentClassifier {
  private val logger = LoggerFactory.getLogger(classOf[CnnSentimentClassifier])
}

class CnnSentimentClassifier extends AutoCloseable {
    private var graph = new Graph
private var textModel: TextModel = null
  @throws[IOException]
    def load_model(inputStream:InputStream): Unit = {
        val bytes = InputStreamUtils.getBytes(inputStream)
        graph.importGraphDef(bytes)
    }

    def load_vocab(inputStream: InputStream): Unit = {
      textModel = ResourceUtils.getTextModel(inputStream)
    }

    def predict(text: String): Array[Float] = {

        val textTensor: Tensor[java.lang.Float] = textModel.toTensor(text)
        try {
            val sess = new Session(graph)
            val result = sess.runner.feed("embedding_1_input:0", textTensor).feed("spatial_dropout1d_1/keras_learning_phase:0", Tensor.create(false)).fetch("output_node0:0").run.get(0).expect(classOf[java.lang.Float])
            try {
                val rshape = result.shape
                if (result.numDimensions != 2 || rshape(0) != 1) throw new RuntimeException(String.format("Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s", java.util.Arrays.toString(rshape)))
                val nlabels: Int = rshape(1).toInt
                val predicted = result.copyTo(Array.ofDim[Float](1, nlabels))(0)
                return predicted
            } catch {
                case ex: Exception =>
                    CnnSentimentClassifier.logger.error("Failed to predict image", ex)
            } finally {
                if (sess != null) sess.close()
                if (result != null) result.close()
            }
        }
        Array.ofDim[Float](2)
    }

  @throws[Exception]
    override def close(): Unit = if (graph != null) {
        graph.close()
        graph = null
    }
}
