package com.github.chen0040.tensorflow.classifiers.demo

import com.github.chen0040.tensorflow.classifiers.sentiment.BidirectionalLstmSentimentClassifier
import com.github.chen0040.tensorflow.classifiers.utils.ResourceUtils

import scala.collection.JavaConversions._

object BidirectionalLstmSentimentClassifierDemo {
  def main(args: Array[String]): Unit = {
    val classifier = new BidirectionalLstmSentimentClassifier()
    classifier.load_model(ResourceUtils.getInputStream("tf_models/wordvec_bidirectional_lstm.pb"))
    classifier.load_vocab(ResourceUtils.getInputStream("tf_models/wordvec_bidirectional_lstm.csv"))

    val lines = ResourceUtils.getLines("data/umich-sentiment-train.txt")

    for(line <- lines){
      val label = line.split("\t")(0)
      val text = line.split("\t")(1)
      val predicted = classifier.predict(text)
      val predicted_label = classifier.predict_label(text)
      System.out.println(text)
      System.out.println("Outcome: " + predicted(0) + ", " + predicted(1))
      System.out.println("Predicted: " + predicted_label + " Actual: " + label)
    }
  }
}
