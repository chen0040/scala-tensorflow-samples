# scala-tensorflow-samples

Scala samples codes on how to load tensorflow pb model and use them to predict



# Usage

### Image Classification using Cifar10

Below show the [demo codes](image-classifier/src/main/scala/com/github/chen0040/tflite/classifiers/demo/Cifar10ImageClassifierDemo.java)
of the  Cifar10ImageClassifier which loads the [cnn_cifar10.pb](image-classifier/src/main/resources/tf_models/cnn_cifar10.pb)
tensorflow model file, and uses it to do image classification:

```scala
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
```

### Image Classification using Inception 

Below show the [demo codes](image-classifier/src/main/scala/com/github/chen0040/tflite/classifiers/demo/InceptionImageClassifierDemo.java)
of the  InceptionImageClassifier which loads the [tensorflow_inception_graph.pb](image-classifier/src/main/resources/tf_models/tensorflow_inception_graph.pb)
tensorflow model file, and uses it to do image classification:

```scala
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
    val image_names = Array[String]("tiger", "lion")
    for (image_name <- image_names) {
      val image_path = "images/inception/" + image_name + ".jpg"
      val img = ResourceUtils.getImage(image_path)
      val predicted_label = classifier.predict_image(img)
      System.out.println("predicted class for " + image_name + ": " + predicted_label)
    }
  }
}
```
