package com.github.chen0040.tensorflow.classifiers.utils

import java.io.{BufferedReader, IOException, InputStream, InputStreamReader}
import java.util
import java.util.stream.{Collector, Collectors}

import org.slf4j.{Logger, LoggerFactory}

import scala.language.implicitConversions
import scala.collection.JavaConversions._

class ResourceUtils {

}

object ResourceUtils {
  def getInputStream(file_path: String): InputStream = classOf[ResourceUtils].getClassLoader.getResourceAsStream(file_path)
  def logger : Logger = LoggerFactory.getLogger(classOf[ResourceUtils])

  @throws[IOException]
  def getTextModel(file_path: String): TextModel = {
    getTextModel(getInputStream(file_path))
  }

  @throws[IOException]
  def getTextModel(inputStream: InputStream): TextModel = {
    var maxLen = 0
    val word2idx: util.Map[String, Int] = new util.HashMap[String, Int]()
    val idx2label: util.Map[Int, String] = new util.HashMap[Int, String]()
    try {
      val reader = new BufferedReader(new InputStreamReader(inputStream))
      var firstLine = true
      var line = reader.readLine()

      while(line != null){
        if(firstLine) {
          firstLine = false
          maxLen = Integer.parseInt(line.trim)
        } else {
          if(line.startsWith("label")) {
            val parts: Array[String] = line.trim.split('\t')
            val word = parts(1)
            val index = Integer.parseInt(parts(2))
            idx2label.put(index, word)
          } else {
            val parts: Array[String] = line.trim.split('\t')
            val word = parts(0)
            val index = Integer.parseInt(parts(1))
            word2idx.put(word, index)
          }

        }
        line = reader.readLine()
      }
    } catch{
      case ex: Exception =>
        logger.error("Failed to get text model", ex)
    }
    new TextModel(maxLen, word2idx, idx2label)
  }

  @throws[IOException]
  def getLines(path: String): util.List[String] = {
    val is = getInputStream(path)

    val result = new util.ArrayList[String]()
    val reader = new BufferedReader(new InputStreamReader(is))
    var line = reader.readLine()
    while(line != null) {
      result.add(line)
      line = reader.readLine()
    }
    result
  }

  def main(args: Array[String]): Unit = {
    val model = getTextModel(file_path = "tf_models/lstm_softmax.csv")
    System.out.println("max_len: " + model.maxLen)
    for(entry <- model.word2idx.entrySet()) {
      System.out.println(entry.getKey + ": " + entry.getValue)
    }
    val lines = getLines("data/umich-sentiment-train.txt")
    for(line <- lines) {
      System.out.println(line)
    }
  }
}