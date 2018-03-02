package com.github.chen0040.tensorflow.classifiers.cifar10

import java.awt.image.BufferedImage
import java.io.{IOException, InputStream}
import java.util

import com.github.chen0040.tensorflow.classifiers.utils
import com.github.chen0040.tensorflow.classifiers.utils.{ImageUtils, InputStreamUtils}
import org.slf4j.LoggerFactory
import org.tensorflow.{Graph, Session, Tensor}

object Cifar10ImageClassifier {
  private val labels = Array[String]("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
  private val logger = LoggerFactory.getLogger(classOf[Cifar10ImageClassifier])
}

class Cifar10ImageClassifier() extends AutoCloseable {
  private var graph = new Graph

  @throws[IOException]
  def load_model(inputStream: InputStream): Unit = {
    val bytes = InputStreamUtils.getBytes(inputStream)
    graph.importGraphDef(bytes)
  }

  def predict_image(image: BufferedImage): String = predict_image(image, 32, 32)

  def predict_image(image: BufferedImage, imgWidth: Int, imgHeight: Int): String = {
    val image2 = ImageUtils.resizeImage(image, imgWidth, imgHeight)
    val imageTensor = utils.TensorUtils.getImageTensorScaled(image2, imgWidth, imgHeight)
    try {
      val sess = new Session(graph)
      val result = sess.runner.feed("conv2d_1_input:0", imageTensor).feed("dropout_1/keras_learning_phase:0", Tensor.create(false)).fetch("output_node0:0").run.get(0).expect(classOf[java.lang.Float])
      try {
        val rshape = result.shape
        if (result.numDimensions != 2 || rshape(0) != 1) throw new RuntimeException(String.format("Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s", util.Arrays.toString(rshape)))
        val nlabels: Int = rshape(1).toInt
        val predicted = result.copyTo(Array.ofDim[Float](1, nlabels))(0)
        var argmax = 0
        var max = predicted(0)
        for (i <- 1 until nlabels) {
          if (max < predicted(i)) {
            max = predicted(i)
            argmax = i
          }
        }
        return Cifar10ImageClassifier.labels(argmax)
      } catch {
        case ex: Exception =>
          Cifar10ImageClassifier.logger.error("Failed to predict image", ex)
      } finally {
        if (sess != null) sess.close()
        if (result != null) result.close()
      }
    }
    "unknown"
  }

  @throws[Exception]
  override def close(): Unit = if (graph != null) {
    graph.close()
    graph = null
  }
}