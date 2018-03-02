package com.github.chen0040.tensorflow.classifiers.demo

import java.io.IOException

import com.github.chen0040.tensorflow.classifiers.cifar10.Cifar10ImageClassifier
import com.github.chen0040.tensorflow.classifiers.utils.ResourceUtils
import org.slf4j.LoggerFactory

class Cifar10ImageClassifierDemo() {

}

object Cifar10ImageClassifierDemo {
  private val logger = LoggerFactory.getLogger(classOf[Cifar10ImageClassifierDemo])

  @throws[IOException]
  def main(args: Array[String]): Unit = {
    val inputStream = ResourceUtils.getInputStream("tf_models/cnn_cifar10.pb")
    val classifier = new Cifar10ImageClassifier
    classifier.load_model(inputStream)
    val image_names = Array[String]("airplane1", "airplane2", "airplane3", "automobile1", "automobile2", "automobile3", "bird1", "bird2", "bird3", "cat1", "cat2", "cat3")
    for (image_name <- image_names) {
      val image_path = "images/cifar10/" + image_name + ".png"
      val img = ResourceUtils.getImage(image_path)
      val predicted_label = classifier.predict_image(img)
      System.out.println("predicted class for " + image_name + ": " + predicted_label)
    }
  }
}