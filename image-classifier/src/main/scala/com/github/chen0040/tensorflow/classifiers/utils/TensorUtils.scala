package com.github.chen0040.tensorflow.classifiers.utils

import java.awt.image.BufferedImage
import java.io.{IOException, InputStream}
import java.nio.FloatBuffer
import javax.imageio.ImageIO

import org.tensorflow.Tensor

object TensorUtils {
  def getImageTensorScaled(image: BufferedImage, imgWidth: Int, imgHeight: Int): Tensor[java.lang.Float] = {
    val channels = 3
    // Generate image file to array
    var index = 0
    val fb = FloatBuffer.allocate(imgWidth * imgHeight * channels)
    // Convert image file to multi-dimension array
    for ( row <- 0 until imgHeight) {
      for ( column <- 0 until imgWidth) {
        val pixel = image.getRGB(column, row)
        var red: Float = (pixel >> 16) & 0xff
        var green: Float = (pixel >> 8) & 0xff
        var blue: Float = pixel & 0xff
        red = red / 255.0f
        green = green / 255.0f
        blue = blue / 255.0f
        fb.put(index, red)
        index += 1
        fb.put(index, green)
        index += 1
        fb.put(index, blue)
        index += 1
      }
    }
    Tensor.create(Array[Long](1, imgWidth, imgHeight, channels), fb)
  }

  def getImageTensorNormalized(image: BufferedImage, imgWidth: Int, imgHeight: Int): Tensor[java.lang.Float] = {
    val channels = 3
    // Generate image file to array
    var index = 0
    val fb = FloatBuffer.allocate(imgWidth * imgHeight * channels)
    // Convert image file to multi-dimension array
    for ( row <- 0 until imgHeight) {
      for ( column <- 0 until imgWidth) {
        val pixel = image.getRGB(column, row)
        var red: Float = (pixel >> 16) & 0xff
        var green: Float = (pixel >> 8) & 0xff
        var blue: Float = pixel & 0xff
        red = (red-127.5f) / 127.5f
        green = (green-127.5f) / 127.5f
        blue = (blue-127.5f) / 127.5f
        fb.put(index, red)
        index += 1
        fb.put(index, green)
        index += 1
        fb.put(index, blue)
        index += 1
      }
    }
    Tensor.create(Array[Long](1, imgWidth, imgHeight, channels), fb)
  }

  @throws[IOException]
  def getImageTensorScaled(inputStream: InputStream): Tensor[java.lang.Float] = {
    val img = ImageIO.read(inputStream)
    val imgWidth = img.getWidth
    val imgHeight = img.getHeight
    getImageTensorScaled(img, imgWidth, imgHeight)
  }
}