package com.github.chen0040.tensorflow.classifiers.demo

import com.github.chen0040.tensorflow.classifiers.sentiment.CnnSentimentClassifier
import com.github.chen0040.tensorflow.classifiers.utils.ResourceUtils
import scala.collection.JavaConversions._

object CnnSentimentClassifierDemo {
  def main(args: Array[String]): Unit = {
    val classifier = new CnnSentimentClassifier()
    classifier.load_model(ResourceUtils.getInputStream("tf_models/wordvec_cnn.pb"))
    classifier.load_vocab(ResourceUtils.getInputStream("tf_models/wordvec_cnn.csv"))

    val lines = ResourceUtils.getLines("data/umich-sentiment-train.txt")
    for(line <- lines){
      val text = line.split("\t")(1)
      val predicted = classifier.predict(text)
      System.out.println(text)
      System.out.println("Predicted: " + predicted)
    }
  }
}
