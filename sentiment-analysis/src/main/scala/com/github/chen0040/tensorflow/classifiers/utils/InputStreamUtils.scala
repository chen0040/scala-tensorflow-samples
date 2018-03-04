package com.github.chen0040.tensorflow.classifiers.utils

import scala.language.implicitConversions
import java.io.{ByteArrayOutputStream, IOException, InputStream}

object InputStreamUtils {
  @throws[IOException]
  def getBytes(is: InputStream): Array[Byte] = {
    val mem = new ByteArrayOutputStream
    val buffer = new Array[Byte](1024)
    var len: Int = is.read(buffer, 0, 1024)
    while (len > 0) {
      mem.write(buffer, 0, len)
      len = is.read(buffer, 0, 1024)
    }
    mem.toByteArray
  }
}