package com.github.chen0040.tensorflow.classifiers.utils

import java.awt._
import java.awt.image.BufferedImage

object ImageUtils {
  def resizeImage(img: BufferedImage, imgWidth: Int, imgHeight: Int): BufferedImage = {
    if (img.getWidth != imgWidth || img.getHeight != imgHeight) {
      val newImg = img.getScaledInstance(imgWidth, imgHeight, Image.SCALE_SMOOTH)
      val newBufferedImg = new BufferedImage(newImg.getWidth(null), newImg.getHeight(null), BufferedImage.TYPE_INT_RGB)
      newBufferedImg.getGraphics.drawImage(newImg, 0, 0, null)
      return newBufferedImg
    }
    img
  }
}