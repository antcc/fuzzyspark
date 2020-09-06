/**
 * WM.scala
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

import scala.math.{pow, exp}

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

/**
 * Wang & Mendel fuzzy rule learning algorithm [1].
 *
 * [1] Wang, L. X., & Mendel, J. M. (1992). Generating fuzzy rules
 *     by learning from examples. IEEE Transactions on systems, man,
 *     and cybernetics, 22(6), 1414-1427.
 */
private class WM(
  private var mfType: String
) extends Serializable {

  import WM._

  /**
   *  Find the regions with maximum membership for a given point, in a
   *  specified list of dimensions.
   */
  private def maxRegion(
      z: Vector,
      dimensionRegions: Array[Array[Double]],
      dimensionLimits: Array[(Double, Double)]) = {
    val n = dimensionRegions.size
    var r = List[(Double, Double)]()
    var degree = 1.0

    for (i <- 0 until n) {
      // We have N points: a_0, a_1, ..., a_N-1
      var regions = dimensionRegions(i)
      var limits = dimensionLimits(i)

      // First region: (-inf, a_1)
      var chosenRegion = (Double.NegativeInfinity, regions(1))
      var maxMs = mf(z(i), chosenRegion, limits, mfType)

      // Middle regions: (a_l, a_l+2)
      for (l <- 0 until regions.size - 2) {
        var currRegion = (regions(l), regions(l + 2))
        var currMs = mf(z(i), currRegion, limits, mfType)
        if (currMs > maxMs) {
          maxMs = currMs
          chosenRegion = currRegion
        }
      }

      // Last region: (a_N-2, +inf)
      var lastRegion = (regions(regions.size - 2), Double.PositiveInfinity)
      var lastMs = mf(z(i), lastRegion, limits, mfType)
      if (lastMs > maxMs) {
        maxMs = lastMs
        chosenRegion = lastRegion
      }

      degree = degree * maxMs
      r = chosenRegion :: r
    }

    (r.reverse, degree)
  }

  /**
   * Train a WM model on the given set of points.
   *
   * @note The RDD `data` should be cached for high performance.
   */
  def run(
      data: RDD[(Vector, Vector)],
      numRegions: Array[Int],
      regionLimits: Array[(Double, Double)],
      classification: Boolean = false): WMModel = {
    val sc = data.sparkContext
    val example = data.first()
    val (inputDims, outputDims) = (example._1.size, example._2.size)

    require(
      numRegions.size == inputDims + outputDims,
      "Every dimension must have a desired number of divisions."
    )

    // Separate limits for each dimension
    val limitsInput = regionLimits.slice(0, inputDims)
    val limitsOutput = regionLimits.slice(inputDims, regionLimits.size)

    // Divide each dimension into regions
    val regionsInput
      = computeRegions(numRegions.slice(0, inputDims), limitsInput)
    val regionsOutput
      = computeRegions(numRegions.slice(inputDims, numRegions.size), limitsOutput)

    // Compute fuzzy rule base.
    // Every rule is represented by a collection of input and output regions
    val ruleBase = data.mapPartitions { iterator =>
      for {
        (x, y) <- iterator
      } yield {
        // Compute regions with maximum membership in each dimension
        val (ri, degreeInput) = maxRegion(x, regionsInput, limitsInput)
        val (ro, degreeOutput) = maxRegion(y, regionsOutput, limitsOutput)

        // Compute rule degree
        var degree = degreeInput * degreeOutput

        // Factor in rule weight (currently RW = 1)
        degree = degree * 1.0

        (ri, (ro, degree))
      }
    }.reduceByKey { case (r1, r2) =>
      if (r1._2 > r2._2) r1 else r2
    }.map { case (ri, (ro, _)) => (ri, ro) }.collect.toArray

    WMModel(ruleBase, limitsInput, limitsOutput, mfType)
  }
}

/** Top-level methods for calling Wang-Mendel algorithm. */
object WM {

  /** Possinle MF names */
  val TRIANGULAR = "Triangular"
  val GAUSSIAN = "Gaussian"

  /**
   * Train a Wang-Mendel model using specified parameters.
   *
   * @param data Data points in (input, output) space.
   * @param numRegions Number of regions for each dimension.
   * @param regionLimits Approximate bounds for the points in each dimension.
   * @param mfType Type of membership function to use.
   */
  def train(
      data: RDD[(Vector, Vector)],
      numRegions: Array[Int],
      regionLimits: Array[(Double, Double)],
      mfType: String = TRIANGULAR): WMModel = {
    new WM(mfType).run(data, numRegions, regionLimits)
  }

  /**
   * Compute the different regions in which the
   * input-output space is divided.
   *
   * A region is defined by an interval (a, b) on the extended
   * real line.
   */
  private[frbs] def computeRegions(
    numRegions: Array[Int],
    limits: Array[(Double, Double)]): Array[Array[Double]] = {
    numRegions.zipWithIndex.map { case (n, i) =>
      val (c, d) = limits(i)
      Array.tabulate(2 * n + 1)(j => c + j * ((d - c) / (2 * n)))
    }
  }

  private[frbs] def getGaussianParameters(
      region: (Double, Double),
      regionLimits: (Double, Double)) = {
    val (a, b) = region
    val (lower, upper) = regionLimits

    // First region
    if (a.isNegInfinity) {
      (lower, (b - lower) / 3.0)
    }

    // Last region
    else if (b.isPosInfinity) {
      (upper, (upper - a) / 3.0)
    }

    // Middle regions
    else {
      ((a + b) / 2.0, (b - a) / 6.0)
    }
  }

  /** Compute the membership value of a point to an interval. */
  private[frbs] def mf(
      x: Double,
      region: (Double, Double),
      limits: (Double, Double),
      mfType: String): Double = {
    require(
      mfType == TRIANGULAR || mfType == GAUSSIAN,
      "Membership function must be one of the allowed types."
    )

    val (a, b) = region
    val (lower, upper) = limits

    if (mfType == TRIANGULAR) {
      // First region
      if (a.isNegInfinity) {
        if (x <= lower)
          1.0
        else if ( x > lower && x < b)
          (b - x) / (b - lower)
        else
          0.0
      }

      // Last region
      else if (b.isPosInfinity) {
        if (x >= upper)
          1.0
        else if (x < upper && x > a)
          (x - a) / (upper - a)
        else
          0.0
      }

      // Middle regions
      else {
        val m = (a + b) / 2.0

        if (x > a && x < m)
          (x - a) / (m - a)
        else if (x > m && x < b)
          (b - x) / (b - m)
        else
          0.0
      }
    }

    else if (mfType == GAUSSIAN) {
      // First region
      if (a.isNegInfinity) {
        val s = (b - lower) / 3.0
        if (x <= lower)
          1.0
        else
          math.exp(-math.pow(x - lower, 2) / (2.0 * s * s))
      }

      // Last region
      else if (b.isPosInfinity) {
        val s = (upper - a) / 3.0
        if (x >= upper)
          1.0
        else
          math.exp(-math.pow(x - upper, 2) / (2.0 * s * s))
      }

      // Middle regions
      else {
        val m = (a + b) / 2.0
        val s = (b - a) / 6.0
        math.exp(-math.pow(x - m, 2) / (2.0 * s * s))
      }
    }

    else {  // Wrong mfType
      0.0
    }
  }
}
