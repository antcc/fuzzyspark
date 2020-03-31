/**
 * ModelIdentification.scala
 * Copyright (C) 2020 antcc
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see
 * https://www.gnu.org/licenses/gpl-3.0.html.
 */

package fuzzyspark.clustering

import scala.math.{exp, pow}

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

/**
 * Implementation of Chiu's model identification method [1]
 * applying fuzzy subtractive clustering.
 *
 * [1] Chiu, S. L. (1994). Fuzzy model identification based on
 *     cluster estimation. Journal of Intelligent & fuzzy systems,
 *     2(3), 267-278.
 */
class ModelIdentification(
  var centers: Array[Vector],
  var ra: Double,
  var inputDims: Int
) extends Serializable {

  /** Rest of algorithm parameters */
  private var alpha = 4 / (ra * ra)
  private var dims = centers(0).size

  /** Get learned centroids for this model. */
  def getCenters: Array[Vector] = centers

  /** Get effective neighbourhood radius. */
  def getRadius: Double = ra

  /** Get number of input dimensions for this model. */
  def getInputDims: Int = inputDims

  /**
   * Predict output values from  a collection of input vectors.
   *
   * @return An RDD of pairs (input, output).
   */
  def predict(input: RDD[Vector]): RDD[(Vector, Vector)] = {
    val outputDims = dims - inputDims
    var centersBroadcast = input.sparkContext.broadcast(centers)

    // Compute output values for each input vector
    val output = input.map { y =>
      // Compute degree of membership of input vector to each cluster
      val membership = centersBroadcast.value.map { c =>
        val inputSlice = Vectors.dense(c.toArray.slice(0, inputDims))
        exp(-alpha * Vectors.sqdist(y, inputSlice))
      }

      // Compute output values
      var z = centersBroadcast.value.zip(membership).map { case (c, mu) =>
        val outputSlice = Vectors.dense(c.toArray.slice(inputDims, dims))
        VectorUtils.multiply(outputSlice, mu)
      }.reduce ( VectorUtils.add )

      z = VectorUtils.divide(z, membership.sum)

      // Form a pair (input, output)
      (y, z)
    }

    output
  }
}

/** Top-level Model Identification methods. */
object ModelIdentification {

  /**
   * Construct a model with specified parameters.
   *
   * @param centers Cluster centers obtained from subtractive clustering.
   * @param ra Neighbourhood radius.
   * @param inputDims Number of input dimensions.
   */
  def apply(
    centers: Array[Vector],
    ra: Double,
    inputDims: Int): ModelIdentification =
    new ModelIdentification(
      centers,
      ra,
      inputDims)
}
