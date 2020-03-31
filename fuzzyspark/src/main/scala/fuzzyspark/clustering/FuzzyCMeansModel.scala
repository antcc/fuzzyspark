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
  var trainingIter: Int
) extends Serializable {

  /** Number of cluster centers. */
  def c: Int = clusterCenters.length

  /** Compute loss function using given data. */
  def computeLoss(data: RDD[Vector]): Double = {
    val sc = data.sparkContext
    val u = membershipMatrix(data)
    FuzzyCMeans.computeLoss(u, sc.broadcast(clusterCenters), m)
  }

  /** Compute fuzzy membership matrix for given data. */
  private def membershipMatrix(data: RDD[Vector]) = {
    val sc = data.sparkContext
    data.map ( FuzzyCMeans.computeRow(_, sc.broadcast(clusterCenters), m) )
  }

  // TODO: predict and fuzzyPredict (by point and by RDD)
  // TODO: fuzzy partition coefficient (https://github.com/scikit-fuzzy/scikit-fuzzy/blob/master/skfuzzy/cluster/_cmeans.py#L101)
}

/** Factory for [[fuzzyspark.clustering.FuzzyCMeansModel]] instances. */
object FuzzyCMeansModel {

  /** Construct a FuzzyCMeansModel resulting from training. */
  def apply(
    clusterCenters: Array[Vector],
    m: Int,
    trainingLoss: Double,
    trainingIter: Int): FuzzyCMeansModel = {
    new FuzzyCMeansModel(clusterCenters, m, trainingLoss, trainingIter)
  }

  /**
   * Construct a FuzzyCMeansModel from given cluster centers and
   * fuzziness degree. A value of 0 for trainingLoss and trainingIter
   * means that the model hasn't been trained.
   */
  def apply(clusterCenters: Array[Vector], m: Int): FuzzyCMeansModel = {
    new FuzzyCMeansModel(clusterCenters, m, 0.0, 0)
  }

  //TODO: load and save model
 }
