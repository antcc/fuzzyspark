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

import fuzzyspark.VectorUtils

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
  var labels: Option[Array[Double]]
) extends Serializable {

  /** Rest of algorithm parameters */
  private var alpha = 4 / (ra * ra)
  private var dims = centers(0).size

  /** Get learned centroids for this model. */
  def getCenters: Array[Vector] = centers

  /** Get centroid labels. */
  def getLabels: Option[Array[Double]] = labels

  /** Set centroids labels. */
  def setLabels(labels: Array[Double]): this.type = {
    this.labels = Option(labels)
    this
  }

  /** Get effective neighbourhood radius. */
  def getRadius: Double = ra

  /** Predict output values for an input vector. */
  def predictOutput(y: Vector): Vector = {
    val inputDims = y.size
    val outputDims = dims - inputDims

    // Compute degree of membership of input vector to each cluster
    val membership = centers.map { c =>
      val inputSlice = Vectors.dense(c.toArray.slice(0, inputDims))
      exp(-alpha * Vectors.sqdist(y, inputSlice))
    }

    // Compute output values
    var z = centers.zip(membership).map { case (c, mu) =>
      val outputSlice = Vectors.dense(c.toArray.slice(inputDims, dims))
      VectorUtils.multiply(outputSlice, mu)
    }.reduce ( VectorUtils.add )

    var mSum = membership.sum
    if (mSum <= 0)
      mSum = mSum + 1e4

    VectorUtils.divide(z, mSum)
  }

  /**
   * Predict output values from a collection of input vectors,
   * taking the first `inputDims` dimensions to be the input
   * space on the cluster centers, and the rest to be the output space.
   *
   * @return An RDD of pairs (input, output).
   */
  def predictOutputRDD(input: RDD[Vector]): RDD[(Vector, Vector)] = {
    val inputDims = input.take(1).size
    val outputDims = dims - inputDims
    val centersBroadcast = input.sparkContext.broadcast(centers)

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

      var mSum = membership.sum
      if (mSum <= 0)
        mSum = mSum + 1e4

      z = VectorUtils.divide(z, mSum)

      // Form a pair (input, output)
      (y, z)
    }

    output
  }

  /** Predict label for a single point. */
  def predictLabel(x: Vector): Double = {
    require(
      labels.exists( _.size == centers.size ),
      s"Each cluster center must have a label.")

    val membership = Vectors.dense(
      centers.map ( c => exp(-alpha * Vectors.sqdist(x, c)) ))
    labels.get(membership.argmax)
  }

  /** Predict labels for unseen data points. */
  def predictLabelsRDD(input: RDD[Vector]): RDD[(Vector, Double)] = {
    require(
      labels.exists( _.size == centers.size ),
      s"Each cluster center must have a label.")

    val centersBroadcast = input.sparkContext.broadcast(centers)
    val output = input.map { x =>
      val membership = Vectors.dense(
        centersBroadcast.value.map ( c => exp(-alpha * Vectors.sqdist(x, c)) ))
      (x, labels.get(membership.argmax))
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
   * @param labels Labels for cluster centers
   */
  def apply(
    centers: Array[Vector],
    ra: Double,
    labels: Option[Array[Double]] = None): ModelIdentification =
    new ModelIdentification(
      centers,
      ra,
      labels)
}
