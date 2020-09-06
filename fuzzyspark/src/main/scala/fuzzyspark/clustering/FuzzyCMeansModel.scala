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

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD

/** A clustering model for Fuzzy C Means. */
class FuzzyCMeansModel(
  var clusterCenters: Array[Vector],
  var m: Double,
  var trainingIter: Int,
  var labels: Option[Array[Double]]
) extends Serializable {

  /** Number of cluster centers. */
  def c: Int = clusterCenters.length

  /** Compute loss function for a given set of points. */
  def computeLoss(data: RDD[Vector]): Double = {
    FuzzyCMeans.computeLoss(data, clusterCenters, m)
  }

  /** Compute fuzzy membership matrix for a given set of points. */
  def membershipMatrix(data: RDD[Vector]): RowMatrix = {
    val sc = data.sparkContext
    val centersBc = sc.broadcast(clusterCenters)
    val u = data.map ( x => Vectors.dense(FuzzyCMeans.computeMembershipRow(x, centersBc.value, m)) )

    new RowMatrix(u)
  }

  /** Compute labels for centroids based on a given set of points. */
  def computeLabels(
      labeledData: RDD[(Vector, Double)],
      alpha: Double,
      scaleAlpha: Boolean = true): Array[Double] = {
    FuzzyCMeans.computeCentersLabels(
      labeledData, clusterCenters,
      m, alpha, scaleAlpha)
  }

  /** Predict point label. */
  def predict(x: Vector): Double = {
    require(
      labels.exists( _.size == clusterCenters.size ),
      s"Each cluster center must have a label.")

    val r = FuzzyCMeans.computeMembershipRow(x, clusterCenters, m)
    val index = r.indexOf(r.max)
    labels.get(index)
  }

  /**
   * Predict labels for unseen data using FCM
   * membership function.
   */
  def predictRDD(data: RDD[Vector]): RDD[Double] = {
    require(
      labels.exists( _.size == clusterCenters.size ),
      s"Each cluster center must have a label.")

    data.map ( x => predict(x) )
  }
}

/** Factory for [[fuzzyspark.clustering.FuzzyCMeansModel]] instances. */
object FuzzyCMeansModel {

  /** Construct a FuzzyCMeansModel resulting from training. */
  def apply(
    clusterCenters: Array[Vector],
    m: Double,
    trainingIter: Int,
    labels: Option[Array[Double]] = None): FuzzyCMeansModel = {
    new FuzzyCMeansModel(clusterCenters, m, trainingIter, labels)
  }

  /**
   * Construct a FuzzyCMeansModel from given cluster centers and
   * fuzziness degree. A value of 0 for trainingIter means that
   * the model training history is unknown.
   */
  def apply(
    clusterCenters: Array[Vector],
    m: Double,
    labels: Option[Array[Double]]): FuzzyCMeansModel = {
    new FuzzyCMeansModel(clusterCenters, m, 0, labels)
  }
 }
