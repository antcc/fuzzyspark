/**
 * FuzzyCMeansModel.scala
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

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

/** A clustering model for Fuzzy C Means. */
class FuzzyCMeansModel(
  var clusterCenters: Array[Vector],
  var m: Int,
  var trainingLoss: Double,
  var trainingIter: Int,
  var labels: Option[Array[Double]]
) extends Serializable {

  /** Get centroid labels. */
  def getLabels: Option[Array[Double]] = labels

  /** Set centroids labels. */
  def setLabels(labels: Array[Double]): this.type = {
    this.labels = Option(labels)
    this
  }

  /** Number of cluster centers. */
  def c: Int = clusterCenters.length

  /** Compute loss function using given data. */
  def computeLoss(data: RDD[Vector]): Double = {
    val centersBroadcast = data.sparkContext.broadcast(clusterCenters)
    val u = membershipMatrix(data)
    FuzzyCMeans.computeLoss(u, centersBroadcast.value, m)
  }

  /** Compute fuzzy membership matrix for given data. */
  private def membershipMatrix(data: RDD[Vector]) = {
    val centersBroadcast = data.sparkContext.broadcast(clusterCenters)
    // Label 0 as a placeholder
    data.map ( x => ((x, 0.0), FuzzyCMeans.computeRow(x, centersBroadcast.value, m)) )
  }

  /**
   * Predict point label
   *
   */
  def predict(x: Vector): Double = {
    require(
      labels.exists( _.size == clusterCenters.size ),
      s"Each cluster center must have a label.")

    val index = FuzzyCMeans.computeRow(x, clusterCenters, m).argmax
    labels.get(index)
  }

  /**
   * Predict labels for unseen data using FCM
   * membership function.
   */
  def predictRDD(data: RDD[Vector]): RDD[(Vector, Double)] = {
    require(
      labels.exists( _.size == clusterCenters.size ),
      s"Each cluster center must have a label.")

    val centersBroadcast = data.sparkContext.broadcast(clusterCenters)
    data.map { x =>
      val index = FuzzyCMeans.computeRow(x, centersBroadcast.value, m).argmax
      (x, labels.get(index))
    }
  }
}

/** Factory for [[fuzzyspark.clustering.FuzzyCMeansModel]] instances. */
object FuzzyCMeansModel {

  /** Construct a FuzzyCMeansModel resulting from training. */
  def apply(
    clusterCenters: Array[Vector],
    m: Int,
    trainingLoss: Double,
    trainingIter: Int,
    labels: Option[Array[Double]] = None): FuzzyCMeansModel = {
    new FuzzyCMeansModel(clusterCenters, m, trainingLoss, trainingIter, labels)
  }

  /**
   * Construct a FuzzyCMeansModel from given cluster centers and
   * fuzziness degree. A value of 0 for trainingLoss and trainingIter
   * means that the model training history is unknown.
   */
  def apply(
    clusterCenters: Array[Vector],
    m: Int,
    labels: Option[Array[Double]]): FuzzyCMeansModel = {
    new FuzzyCMeansModel(clusterCenters, m, 0.0, 0, labels)
  }

  //TODO: load and save model
 }
