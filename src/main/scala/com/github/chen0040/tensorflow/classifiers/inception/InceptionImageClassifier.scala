package com.github.chen0040.tensorflow.classifiers.inception

import java.awt.image.BufferedImage
import java.io.{BufferedReader, IOException, InputStream, InputStreamReader}
import java.util

import com.github.chen0040.tensorflow.classifiers.utils.{ImageUtils, InputStreamUtils, ResourceUtils, TensorUtils}
import org.slf4j.LoggerFactory
import org.tensorflow.{Graph, Session}

object InceptionImageClassifier {
  private val logger = LoggerFactory.getLogger(classOf[InceptionImageClassifier])
}

class InceptionImageClassifier() extends AutoCloseable {
  private var graph = new Graph
  private val labels = new util.ArrayList[String]

  @throws[IOException]
  def load_model(): Unit = {
    var inputStream = ResourceUtils.getInputStream("tf_models/tensorflow_inception_graph.pb")
    load_model(inputStream)
    inputStream = ResourceUtils.getInputStream("tf_models/imagenet_comp_graph_label_strings.txt")
    load_labels(inputStream)
  }

  @throws[IOException]
  def load_model(inputStream: InputStream): Unit = {
    val bytes = InputStreamUtils.getBytes(inputStream)
    graph.importGraphDef(bytes)
  }

  def load_labels(inputStream: InputStream): Unit = {
    labels.clear()
    try {
      val reader = new BufferedReader(new InputStreamReader(inputStream))
      try {
        var line = reader.readLine
        while (line != null){
          labels.add(line)
          line = reader.readLine
        }
      } catch {
        case e: IOException =>
          e.printStackTrace()
      } finally if (reader != null) reader.close()
    }
  }

  def predict_image(image: BufferedImage): String = predict_image(image, 224, 224)

  def predict_image(image: BufferedImage, imgWidth: Int, imgHeight: Int): String = {
    val image2 = ImageUtils.resizeImage(image, imgWidth, imgHeight)
    val imageTensor = TensorUtils.getImageTensor(image, imgWidth, imgHeight)
    try {
      val sess = new Session(graph)
      val result = sess.runner.feed("input", imageTensor).fetch("output").run.get(0).expect(classOf[java.lang.Float])
      try {
        val rshape = result.shape
        if (result.numDimensions != 2 || rshape(0) != 1) throw new RuntimeException(String.format("Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s", util.Arrays.toString(rshape)))
        val nlabels = rshape(1).toInt
        val predicted = result.copyTo(Array.ofDim[Float](1, nlabels))(0)
        var argmax = 0
        var max = predicted(0)
        for (i <- 1 until nlabels) {
          if (max < predicted(i)) {
            max = predicted(i)
            argmax = i
          }
        }
        if (argmax >= 0 && argmax < labels.size) return labels.get(argmax)
        else return "unknown"
      } catch {
        case ex: Exception =>
          InceptionImageClassifier.logger.error("Failed to predict image", ex)
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