package com.github.chen0040.tensorflow.classifiers.utils
import java.nio.{FloatBuffer, IntBuffer}

import org.tensorflow.Tensor

import scala.language.implicitConversions

class TextModel(val maxLen: Int, val word2idx: java.util.Map[String, Int]) {
  def toTensor(text: String): Tensor[java.lang.Integer] = {
    val ib = IntBuffer.allocate(maxLen)

    var index = 0
    for(word: String <- text.toLowerCase().split(" ")){
      var idx = 0
      if(word2idx.containsKey(word)) {
        idx  = word2idx.get(word)
      }
      ib.put(index, idx)
      index += 1
    }

    Tensor.create(Array[Long](1, maxLen), ib)

  }
}
