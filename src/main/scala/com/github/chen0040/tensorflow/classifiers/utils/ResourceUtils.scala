package com.github.chen0040.tensorflow.classifiers.utils

import java.awt.image.BufferedImage
import java.io.{IOException, InputStream}
import javax.imageio.ImageIO

class ResourceUtils {

}

object ResourceUtils {
  def getInputStream(file_path: String): InputStream = classOf[ResourceUtils].getClassLoader.getResourceAsStream(file_path)

  @throws[IOException]
  def getImage(file_path: String): BufferedImage = ImageIO.read(getInputStream(file_path))
}