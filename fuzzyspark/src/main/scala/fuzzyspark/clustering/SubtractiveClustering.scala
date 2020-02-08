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

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

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
   *   { ra = 0.3, lowerBound = 0.15, upperBound = 0.5,
   *     numPartitions = 8 }
   */
  def this() = this(0.3, 0.15, 0.5, 8)

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
    val numPoints = data.count()
    var centers = List[Vector]()

    // Compute initial potential
    val pairs = data.cartesian(data)
    var potential = pairs.map { case (a,b) =>
      (a, math.exp(-alpha * Vectors.sqdist(a, b)))
    }.reduceByKey ( _ + _ ).cache()

    var chosenTuple = potential.max()(Ordering[Double].on ( x => x._2 ))
    var chosenCenter = chosenTuple._1
    var firstCenterPotential, chosenPotential = chosenTuple._2

    // First center
    centers = chosenCenter :: centers

    var stop = false
    var test = true
    while (!stop) {
      // Revise potential of points
      potential = potential.map { case (a,b) =>
        (a, b - chosenPotential *
          math.exp(-beta * Vectors.sqdist(a, chosenCenter)))
      }.cache()

      // Find new center
      chosenTuple = potential.max()(Ordering[Double].on ( x => x._2 ))
      chosenCenter = chosenTuple._1
      chosenPotential = chosenTuple._2
      test = true

      // Check stopping condition
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
          var dmin = centers.map { x =>
            sqrt(Vectors.sqdist(chosenCenter, x))
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
            potential = potential.map { case (a,b) =>
              (a, if (a == chosenCenter) 0.0 else b)
            }.cache()

            // Find new center
            chosenTuple = potential.max()(Ordering[Double].on ( x => x._2 ))
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
    var centersWithIndex = List[(Int, Vector)]()
    var potential = List[(Vector, Double)]()
    var numPoints = 0
    var (it1, it2) = it.duplicate

    while (it1.hasNext) {
      var dup = it2.duplicate
      it2 = dup._1
      var it3 = dup._2
      var cur = it1.next
      var dist = 0.0

      while(it3.hasNext) {
        var aux: Vector = it3.next
        dist = dist + math.exp(-alpha * Vectors.sqdist(cur, aux))
      }

      potential = (cur, dist) :: potential
      numPoints = numPoints + 1
    }

    var chosenTuple = potential.maxBy { _._2 }
    var chosenCenter = chosenTuple._1
    var firstCenterPotential, chosenPotential = chosenTuple._2

    // First center
    centersWithIndex = (index, chosenCenter) :: centersWithIndex

    var stop = false
    var test = true
    while (!stop) {
      // Revise potential of points
      potential = potential.map { case (a,b) =>
        (a, b - chosenPotential *
          math.exp(-beta * Vectors.sqdist(a, chosenCenter)))
      }

      // Find new center
      chosenTuple = potential.maxBy { _._2 }
      chosenCenter = chosenTuple._1
      chosenPotential = chosenTuple._2
      test = true

      // Check stopping condition
      while (test) {
        // Accept and continue
        if (chosenPotential > upperBound * firstCenterPotential) {
          centersWithIndex = (index, chosenCenter) :: centersWithIndex
          test = false
          if (centersWithIndex.length >= numPoints)
            stop = true
        }

        // Reject and stop
        else if (chosenPotential < lowerBound * firstCenterPotential) {
          test = false
          stop = true
        }

        // Gray zone
        else {
          var dmin = centersWithIndex.map { case (_, c) =>
            sqrt(Vectors.sqdist(chosenCenter, c))
          }.reduceLeft ( _ min _ )

          // Accept and continue
          if ((dmin / ra) + (chosenPotential / firstCenterPotential) >= 1) {
            centersWithIndex = (index, chosenCenter) :: centersWithIndex
            test = false
            if (centersWithIndex.length >= numPoints)
              stop = true
          }

          // Reject and re-test
          else {
            potential = potential.map { case (a,b) =>
              (a, if (a == chosenCenter) 0.0 else b)
            }

            // Find new center
            chosenTuple = potential.maxBy { _._2 }
            chosenCenter = chosenTuple._1
            chosenPotential = chosenTuple._2
          }
        }
      }
    }

    centersWithIndex.iterator
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
  def apply(): SubtractiveClustering =
    new SubtractiveClustering()

}
