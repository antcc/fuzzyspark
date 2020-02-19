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
  var numPartitions: Int,
  var numPartitionsPerGroup: Int
) extends Serializable {

  /** Rest of algorithm parameters. */
  private var rb = 1.5 * ra
  private var alpha = 4 / (ra * ra)
  private var beta = 4 / (rb * rb)

  /**
   * Construct object with default parameters:
   *   { ra = 0.3, lowerBound = 0.15,
   *     upperBound = 0.5, numPartitionsPerGroup = numPartitions / 2 }
   */
  def this(numPartitions: Int) = this(0.3, 0.15, 0.5, numPartitions, numPartitions / 2)

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
  def chiuGlobal(
    data: RDD[Vector],
    initPotential: Option[RDD[(Vector, Double)]] = None): Array[Vector] = {
    val sc = data.sparkContext
    val numPoints = data.count()
    var centers = List[Vector]()

    // Compute initial potential
    var potential = initPotential.getOrElse(initPotentialRDD(data))

    // Compute initial center
    var chosenTuple = potential.max()(Ordering[Double].on ( _._2 ))
    var chosenCenter = chosenTuple._1
    var chosenPotential = chosenTuple._2
    val firstCenterPotential = chosenTuple._2
    centers ::= chosenCenter

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
          centers ::= chosenCenter
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
            centers ::= chosenCenter
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
   * subtractive clustering algorithm. The data is spread evenly
   * across partitions and cluster centers are calculated in each one
   * of them.
   *
   * @return Iterator to a List[(Int, Vector, Double)] which contains the
   * cluster centers together with the index of the partition and their
   * potential value.
   */
  def chiuLocal(index: Int, it: Iterator[Vector]): Iterator[(Int, Vector, Double)] = {
    var centersIndexed = List[(Int, Vector, Double)]()

    // Compute initial potential
    var potential: Map[Vector, Double] = initPotentialIterator(it)
    var numPoints = potential.size

    // Compute initial center
    var chosenTuple = potential.maxBy ( _._2 )
    var chosenCenter = chosenTuple._1
    var chosenPotential = chosenTuple._2
    val firstCenterPotential = chosenTuple._2
    centersIndexed ::= (index, chosenCenter, chosenPotential)

    var stop = false
    while (!stop) {
      // Revise potential of points
      potential = potential.map { case (x, y) =>
        (x, y - chosenPotential *
          exp(-beta * Vectors.sqdist(x, chosenCenter)))
      }

      // Find new center
      chosenTuple = potential.maxBy ( _._2 )
      chosenCenter = chosenTuple._1
      chosenPotential = chosenTuple._2

      // Check stopping condition
      var test = true
      while (test) {
        // Accept and continue
        if (chosenPotential > upperBound * firstCenterPotential) {
          centersIndexed ::= (index, chosenCenter, chosenPotential)
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
          var dmin = centersIndexed.map { case (_, c, _) =>
            sqrt(Vectors.sqdist(chosenCenter, c))
          }.reduceLeft ( _ min _ )

          // Accept and continue
          if ((dmin / ra) + (chosenPotential / firstCenterPotential) >= 1) {
            centersIndexed ::= (index, chosenCenter, chosenPotential)
            test = false
            if (centersIndexed.length >= numPoints)
              stop = true
          }
          // Reject and re-test
          else {
            potential = potential + (chosenCenter -> 0.0)

            // Find new center
            chosenTuple = potential.maxBy ( _._2 )
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
   */
  def chiuIntermediate(data: RDD[Vector]): Array[Vector] = {
    val sc = data.sparkContext

    // Get centers per partition
    var localCentersIndexed = data.mapPartitionsWithIndex ( chiuLocal )
      .collect()
      .toArray
    val localCenters = localCentersIndexed.map ( _._2 )
    val numCenters = localCenters.size

    // Refine centers with Fuzzy C Means for a few iterations, using all the data
    /**val fcmModel = FuzzyCMeans.train(
      data,
      initMode = FuzzyCMeans.PROVIDED,
      numPartitions = numPartitions,
      initCenters = Option(localCenters),
      maxIter = 10
    )

    // Update indexed centers
    // NOTE: we don't change the potential even though the centers might have changed
    localCentersIndexed = localCentersIndexed.zipWithIndex.map {
      case ((p, _, d), i) => (p, fcmModel.clusterCenters(i), d)
    }*/

    // Group centers every few partitions
    var centersPotential = List[Array[(Vector, Double)]]()
    for (i <- 0 until numPartitions by numPartitionsPerGroup) {
      var centersGrouped = Array[(Vector, Double)]()
      for (j <- i until i + numPartitionsPerGroup if j < numPartitions) {
        // Normalize potential of centers so that they are comparable
        val centersFiltered = localCentersIndexed.filter ( _._1 == j )
        centersGrouped ++= centersFiltered.map { case (_, c, d) =>
          (c, d / centersFiltered.size)
        }
      }
      centersPotential ::= centersGrouped
    }

    // Apply global version to refine centers in every group and concatenate the results
    var centers = Array[Vector]()
    val numPartitionsOld = numPartitions
    numPartitions = sqrt(numPartitionsOld).toInt
    for (cs <- centersPotential) {
      centers ++= chiuGlobal(
        sc.parallelize(cs.map ( _._1 ), numPartitions),
        Option(sc.parallelize(cs, numPartitions))
      )
    }
    numPartitions = numPartitionsOld

    centers
  }

  /** Compute initial potential of points given as an `RDD`. */
  private def initPotentialRDD(data: RDD[Vector]): RDD[(Vector, Double)] = {
    val pairs = data.cartesian(data)
    val potential = pairs.map { case (x, y) =>
      (x, exp(-alpha * Vectors.sqdist(x, y)))
    }.reduceByKey ( _ + _ ).cache()
    potential
  }

  /** Compute initial potential of points given as an `Iterator`. */
  private def initPotentialIterator(it: Iterator[Vector]): Map[Vector, Double] = {
    val data = it.toArray
    val pairs = data.flatMap ( x => data.map ( y => (x, y) ) )
    val potential = pairs.groupBy ( _._1 ).map { case (x, xs) =>
      val pot = xs.map ( y => exp(-alpha * Vectors.sqdist(x, y._2)) ).sum
      (x, pot)
    }
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
   * @param numPartitionsPerGroup Number of partition per group for intermediate version.
   */
  def apply(
    ra: Double,
    lowerBound: Double,
    upperBound: Double,
    numPartitions: Int,
    numPartitionsPerGroup: Int): SubtractiveClustering =
    new SubtractiveClustering(
      ra,
      lowerBound,
      upperBound,
      numPartitions,
      numPartitionsPerGroup)

  /** Construct a SubtractiveClustering model with default parameters. */
  def apply(numPartitions: Int): SubtractiveClustering =
    new SubtractiveClustering(numPartitions)
}
