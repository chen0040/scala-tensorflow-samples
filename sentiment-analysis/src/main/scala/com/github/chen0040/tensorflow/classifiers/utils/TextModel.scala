package com.github.chen0040.tensorflow.classifiers.utils
import java.nio.{FloatBuffer, IntBuffer}

import org.tensorflow.Tensor

import scala.language.implicitConversions

class TextModel(val maxLen: Int, val word2idx: java.util.Map[String, Int], val idx2label: java.util.Map[Int, String]) {
  def toTensor(text: String): Tensor[java.lang.Float] = {
    val ib = FloatBuffer.allocate(maxLen)

    var index = 0
    val parts = text.toLowerCase().split(" ")
    val textLen = Math.min(maxLen, parts.length)
    for(word: String <- parts){
      var idx = 0
      if(word2idx.containsKey(word)) {
        idx  = word2idx.get(word)
      }
      ib.put(maxLen - textLen + index, idx)
      index += 1
    }

    Tensor.create(Array[Long](1, maxLen), ib)

  }

  def toLabel(idx: Int) : String = idx2label.getOrDefault(idx, "")
}
