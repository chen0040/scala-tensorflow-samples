package com.github.chen0040.tensorflow.classifiers.demo

import java.io.IOException

import com.github.chen0040.tensorflow.classifiers.inception.InceptionImageClassifier
import com.github.chen0040.tensorflow.classifiers.utils.ResourceUtils
import org.slf4j.LoggerFactory

class InceptionImageClassifierDemo {

}

object InceptionImageClassifierDemo {
  private val logger = LoggerFactory.getLogger(classOf[InceptionImageClassifierDemo])

  @throws[IOException]
  def main(args: Array[String]): Unit = {
    val classifier = new InceptionImageClassifier
    classifier.load_model(ResourceUtils.getInputStream("tf_models/tensorflow_inception_graph.pb"))
    classifier.load_labels(ResourceUtils.getInputStream("tf_models/imagenet_comp_graph_label_strings.txt"))
    val image_names = Array[String]("tiger", "lion", "eagle")
    for (image_name <- image_names) {
      val image_path = "images/inception/" + image_name + ".jpg"
      val img = ResourceUtils.getImage(image_path)
      val predicted_label = classifier.predict_image(img)
      System.out.println("predicted class for " + image_name + ": " + predicted_label)
    }
  }
}