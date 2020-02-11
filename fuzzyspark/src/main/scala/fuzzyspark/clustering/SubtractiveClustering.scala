/**
 * SubtractiveClustering.scala
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

import scala.math.{exp, pow, sqrt}

import org.apache.spark.HashPartitioner
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

/**
 * Implementation of Chiu's fuzzy subtractive clustering
 * algorithm [1] for computing initial cluster centers.
 *
 * [1] Chiu, S. L. (1994). Fuzzy model identification based on
 *     cluster estimation. Journal of Intelligent & fuzzy systems,
 *     2(3), 267-278.
 */
class SubtractiveClustering(
  private var ra: Double,
  var lowerBound: Double,
  var upperBound: Double,
  var numPartitions: Int
) extends Serializable {

  /** Rest of algorithm parameters. */
  private var rb = 1.5 * ra
  private var alpha = 4 / (ra * ra)
  private var beta = 4 / (rb * rb)

  /**
   * Construct object with default parameters:
   *   { ra = 0.3, lowerBound = 0.15, upperBound = 0.5 }
   */
  def this(numPartitions: Int) = this(0.3, 0.15, 0.5, numPartitions)

  /** Neighbourhood radius. */
  def getRadius: Double = ra

  /**
   * Set neighbourhood radius and modify the rest of the
   * parameters accordingly.
   */
  def setRadius(ra: Double): this.type = {
    this.ra = ra
    this.rb = 1.5 * ra
    this.alpha = 4 / (ra * ra)
    this.beta = 4 / (rb * rb)
    this
  }

  /**
   * Get cluster centers applying the global version of the
   * subtractive clustering algorithm.
   */
  def chiuGlobal(data: RDD[Vector]): Array[Vector] = {
    val sc = data.sparkContext
    val numPoints = data.count()
    var centers = List[Vector]()

    // Compute initial potential
    var potential: RDD[(Vector, Double)] = initPotential(data)

    // Compute initial center
    var chosenTuple = potential.max()(Ordering[Double].on ( _._2 ))
    var chosenCenter = chosenTuple._1
    var chosenPotential = chosenTuple._2
    val firstCenterPotential = chosenTuple._2
    centers = chosenCenter :: centers

    // Main loop of the algorithm
    var stop = false
    while (!stop) {
      // Revise potential of points
      potential = potential.map { case (x, p) =>
        (x, p - chosenPotential *
          exp(-beta * Vectors.sqdist(x, chosenCenter)))
      }.cache()

      // Find new center
      chosenTuple = potential.max()(Ordering[Double].on ( _._2 ))
      chosenCenter = chosenTuple._1
      chosenPotential = chosenTuple._2

      // Check stopping condition
      var test = true
      while (test) {
        // Accept and continue
        if (chosenPotential > upperBound * firstCenterPotential) {
          centers = chosenCenter :: centers
          test = false
          if (centers.length >= numPoints)
            stop = true
        }
        // Reject and stop
        else if (chosenPotential < lowerBound * firstCenterPotential) {
          test = false
          stop = true
        }
        // Gray zone
        else {
          var dmin = centers.map { c =>
            sqrt(Vectors.sqdist(chosenCenter, c))
          }.reduceLeft ( _ min _ )

          // Accept and continue
          if ((dmin / ra) + (chosenPotential / firstCenterPotential) >= 1) {
            centers = chosenCenter :: centers
            test = false
            if (centers.length >= numPoints)
              stop = true
          }
          // Reject and re-test
          else {
            potential = potential.map { case (x, p) =>
              (x, if (x == chosenCenter) 0.0 else p)
            }.cache()

            // Find new center
            chosenTuple = potential.max()(Ordering[Double].on ( _._2 ))
            chosenCenter = chosenTuple._1
            chosenPotential = chosenTuple._2
          }
        }
      }
    }

    centers.toArray
  }

  /**
   * Get cluster centers applying the local version of the
   * subtractive clustering algorithm.
   *
   * The data is spread evenly across partitions and cluster
   * centers are calculated in each one of them.
   *
   * @return Iterator to a List[(Int, Vector)] which contains the
   * cluster centers together with the index of the partition.
   */
  def chiuLocal(index: Int, it: Iterator[Vector]): Iterator[(Int, Vector)] = {
    var centersIndexed = List[(Int, Vector)]()
    var potential = List[(Vector, Double)]()
    var numPoints = 0
    var (it1, it2) = it.duplicate

    // Compute initial potential
    while (it1.hasNext) {
      var dup = it2.duplicate
      it2 = dup._1
      var it3 = dup._2
      var curr = it1.next

      // Get distance from current point to every point
      var dist = 0.0
      while(it3.hasNext) {
        var aux: Vector = it3.next
        dist += exp(-alpha * Vectors.sqdist(curr, aux))
      }

      // Add potential of current point
      potential = (curr, dist) :: potential
      numPoints = numPoints + 1
    }

    // Compute initial center
    var chosenTuple = potential.maxBy ( _._2 )
    var chosenCenter = chosenTuple._1
    var chosenPotential = chosenTuple._2
    val firstCenterPotential = chosenTuple._2
    centersIndexed = (index, chosenCenter) :: centersIndexed

    var stop = false
    while (!stop) {
      // Revise potential of points
      potential = potential.map { case (x, y) =>
        (x, y - chosenPotential *
          exp(-beta * Vectors.sqdist(x, chosenCenter)))
      }

      // Find new center
      chosenTuple = potential.maxBy { _._2 }
      chosenCenter = chosenTuple._1
      chosenPotential = chosenTuple._2

      // Check stopping condition
      var test = true
      while (test) {
        // Accept and continue
        if (chosenPotential > upperBound * firstCenterPotential) {
          centersIndexed = (index, chosenCenter) :: centersIndexed
          test = false
          if (centersIndexed.length >= numPoints)
            stop = true
        }
        // Reject and stop
        else if (chosenPotential < lowerBound * firstCenterPotential) {
          test = false
          stop = true
        }
        // Gray zone
        else {
          var dmin = centersIndexed.map { case (_, c) =>
            sqrt(Vectors.sqdist(chosenCenter, c))
          }.reduceLeft ( _ min _ )

          // Accept and continue
          if ((dmin / ra) + (chosenPotential / firstCenterPotential) >= 1) {
            centersIndexed = (index, chosenCenter) :: centersIndexed
            test = false
            if (centersIndexed.length >= numPoints)
              stop = true
          }
          // Reject and re-test
          else {
            potential = potential.map { case (x, y) =>
              (x, if (x == chosenCenter) 0.0 else y)
            }

            // Find new center
            chosenTuple = potential.maxBy { _._2 }
            chosenCenter = chosenTuple._1
            chosenPotential = chosenTuple._2
          }
        }
      }
    }

    centersIndexed.iterator
  }

  /**
   * Get cluster centers applying the intermediate version of the
   * subtractive clustering algorithm.
   *
   * @return Iterator to a List[Vector] containing the cluster centers.
   */
  def chiuIntermediate(index: Int, data: Iterator[Vector]): Iterator[Vector] = {
    val localCenters = chiuLocal(index, data)
    localCenters.map ( _._2 )
  }

  /** Compute initial potential of points. */
  private def initPotential(data: RDD[Vector]): RDD[(Vector, Double)] = {
    val pairs = data.cartesian(data).coalesce(numPartitions)
    val potential = pairs.map { case (x, y) =>
      (x, exp(-alpha * Vectors.sqdist(x, y)))
    }.reduceByKey ( _ + _ ).cache()
    potential
  }
}

/** Top-level Subtractive Clustering methods. */
object SubtractiveClustering {

  /**
   * Construct a SubtractiveClustering model with specified parameters.
   *
   * @param ra Neighbourhood radius.
   * @param lowerBound Lower bound for stopping condition.
   * @param upperBound Upper bound for stopping condition.
   * @param numPartitions Number of partitions for local and intermediate versions.
   */
  def apply(
    ra: Double,
    lowerBound: Double,
    upperBound: Double,
    numPartitions: Int): SubtractiveClustering =
    new SubtractiveClustering(ra, lowerBound, upperBound, numPartitions)

  /** Construct a SubtractiveClustering model with default parameters. */
  def apply(numPartitions: Int): SubtractiveClustering =
    new SubtractiveClustering(numPartitions)
}
