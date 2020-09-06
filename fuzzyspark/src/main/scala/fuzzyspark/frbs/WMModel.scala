/**
 * WMModel.scala
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

package fuzzyspark.frbs

import fuzzyspark.VectorUtils

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

/** A FRBS model for Wang-Mendel algorithm. */
class WMModel(
  val ruleBase: Array[(List[(Double, Double)], List[(Double, Double)])],
  val limitsInput: Array[(Double, Double)],
  val limitsOutput: Array[(Double, Double)],
  val mfType: String
) extends Serializable {

  /** Compute the center of a fuzzy region. */
  private def regionCenter(r: (Double, Double), limits: (Double, Double)) = {
    val (a, b) = (r._1, r._2)

    if (a.isNegInfinity)
      limits._1
    else if (b.isPosInfinity)
      limits._2
    else
      (a + b) / 2.0
  }

  /** Predict point output. */
  def predict(
    x: Vector,
    ruleBaseOption: Option[Array[(List[(Double, Double)], List[(Double, Double)])]]
      = None): Vector = {
    val kb = ruleBaseOption.getOrElse(ruleBase)
    val outputDims = kb(0)._2.size

    // Compute output degree for each rule
    val outputDegree = kb.map { case (ri, ro) =>
      (ro, ri.zipWithIndex.map { case (r, i) =>
        WM.mf(x(i), r, limitsInput(i), mfType)
      }.reduce ( _ * _ ))
    }

    // Defuzzify values for each output dimension
    val y = Array.ofDim[Double](outputDims)
    for (j <- 0 until outputDims) {
      var yy = outputDegree.map { case (ro, mu)
        => mu * regionCenter(ro(j), limitsOutput(j))
      }.reduce ( _ + _ )

      var outSum = outputDegree.map ( _._2 ).sum
      if (outSum <= 0)
        outSum = outSum + 1e4

      y(j) = yy / outSum
    }

    Vectors.dense(y)
  }

  /** Predict output for unseen data. */
  def predictRDD(data: RDD[Vector]): RDD[(Vector, Vector)] = {
    val ruleBaseBroadcast = data.sparkContext.broadcast(ruleBase)
    data.map { x =>
      (x, predict(x, Option(ruleBaseBroadcast.value)))
    }
  }
}

/** Factory for [[fuzzyspark.frbs.WMModel]] instances. */
object WMModel {

  /** Construct a WMModel resulting from training. */
  def apply(
    ruleBase: Array[(List[(Double, Double)], List[(Double, Double)])],
    limitsInput: Array[(Double, Double)],
    limitsOutput: Array[(Double, Double)],
    mfType: String): WMModel = {
    new WMModel(ruleBase, limitsInput, limitsOutput, mfType)
  }
 }
