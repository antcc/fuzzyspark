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

import scala.math.{exp, pow, sqrt, min}

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
  private var rb: Double,
  var lowerBound: Double,
  var upperBound: Double,
  var numPartitions: Int,
  var numPartitionsPerGroup: Int,
  private var raGlobal: Double,
  private var rbGlobal: Double,
  var lowerBoundGlobal: Double,
  var upperBoundGlobal: Double
) extends Serializable {

  /** Rest of algorithm parameters. */
  private var alpha = 4 / (ra * ra)
  private var beta = 4 / (rb * rb)
  private val PARTITION_SIZE = 1000

  /**
   * Construct object with default parameters:
   *   { ra = 0.3, rb = 0.45, lowerBound = 0.15,
   *     upperBound = 0.5, numPartitionsPerGroup = numPartitions / 2,
   *     raGlobal = 0.3, rbGlobal = 0.45, lowerBoundGlobal = 0.15,
   *     upperBoundGlobal = 0.5 }
   */
  def this(numPartitions: Int) =
    this(0.3, 0.45, 0.15, 0.5, numPartitions, numPartitions / 2, 0.3, 0.45, 0.15, 0.5)

  /** Neighbourhood radius. */
  def getRadius: Double = ra

  /**
   * Set neighbourhood radius and modify the rest of the
   * parameters accordingly.
   */
  def setRadius(ra: Double): this.type = {
    this.ra = ra
    this.alpha = 4 / (ra * ra)
    this
  }

  /** Potential-dropping neighbourhood radius. */
  def getRb: Double = rb

  /**
   * Set potential-dropping neighbourhood radius and modify the rest
   * of the parameters accordingly.
   */
  def setRb(rb: Double): this.type = {
    this.rb = rb
    this.beta = 4 / (rb * rb)
    this
  }

  /** Neighbourhood radius for global stage of intermediate version. */
  def getRadiusGlobal: Double = raGlobal

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
    var potential = initPotential.getOrElse(initPotentialRDD(data)).cache()

    // Compute initial center
    var chosenTuple = potential.max()(Ordering[Double].on ( _._2 ))
    var chosenCenter = chosenTuple._1
    var chosenPotential = chosenTuple._2
    val firstCenterPotential = chosenTuple._2
    centers ::= chosenCenter

    // Main loop of the algorithm
    var stop = false
    var it = 0
    while (!stop) {
      // Revise potential of points
      potential = potential.map { case (x, p) =>
        (x, p - chosenPotential *
          exp(-beta * Vectors.sqdist(x, chosenCenter)))
      }.cache()

      if (it % 30 == 0) {
        potential.checkpoint()
        potential.count()
      }

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

      it = it + 1
      //println("[ChiuGlobal] Added center #" + it + " with potential " + chosenPotential)
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

    //println("[ChiuLocal] Total no. of centers: " + localCentersIndexed.size)

    // Group centers every few partitions
    var centersPotential = List[Array[(Vector, Double)]]()
    for (i <- 0 until numPartitions by numPartitionsPerGroup) {
      val centersFiltered = localCentersIndexed.filter { case (j, _, _) =>
        j < min(numPartitions, i + numPartitionsPerGroup) && j >= i
      }
      centersPotential ::= centersFiltered.map { case (_, c, d) =>
        (c, d / centersFiltered.size)
      }

      //println("[ChiuLocal] No. of center for group " + i + ": " + centersFiltered.size)
    }

    // Apply global version to refine centers in every group and concatenate the results
    var centers = Array[Vector]()
    val lowerBoundOld = lowerBound
    val upperBoundOld = upperBound
    val raOld = ra
    val rbOld = rb
    lowerBound = lowerBoundGlobal
    upperBound = upperBoundGlobal
    setRadius(raGlobal)
    setRb(rbGlobal)
    for (cs <- centersPotential) {
      var numPartitionsGlobal = cs.size / PARTITION_SIZE + 1
      centers ++= chiuGlobal(
        sc.parallelize(cs.map ( _._1 ), numPartitionsGlobal).cache(),
        Option(sc.parallelize(cs, numPartitionsGlobal))
      )

      println("[ChiuI] Added centers for one partition.")
    }
    lowerBound = lowerBoundOld
    upperBound = upperBoundOld
    setRadius(raOld)
    setRb(rbOld)

    centers
  }

  /** Compute initial potential of points given as an `RDD`. */
  private def initPotentialRDD(data: RDD[Vector]): RDD[(Vector, Double)] = {
    val pairs = data.cartesian(data)
    val potential = pairs.map { case (x, y) =>
      (x, exp(-alpha * Vectors.sqdist(x, y)))
    }.reduceByKey ( _ + _ )

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
   * @param ra Neighbourhood radius for initial potential.
   * @param rb Neighbourhood radius for decreasing potential.
   * @param lowerBound Lower bound for stopping condition.
   * @param upperBound Upper bound for stopping condition.
   * @param numPartitions Number of partitions for local and intermediate versions.
   * @param numPartitionsPerGroup Number of partition per group for intermediate version.
   * @param raGlobal ra for global stage of intermediate version.
   * @param rbGlobal rb for global stage of intermediate version.
   * @param lowerBoundGlobal Lower bound for global stage of intermediate version.
   * @param upperBoundGlobal Upper bound for global stage of intermediate version.
   */
  def apply(
    ra: Double,
    rb: Double,
    lowerBound: Double,
    upperBound: Double,
    numPartitions: Int,
    numPartitionsPerGroup: Int,
    raGlobal: Double = 0.3,
    rbGlobal: Double = 0.45,
    lowerBoundGlobal: Double = 0.15,
    upperBoundGlobal: Double = 0.5): SubtractiveClustering =
    new SubtractiveClustering(
      ra,
      rb,
      lowerBound,
      upperBound,
      numPartitions,
      numPartitionsPerGroup,
      raGlobal,
      rbGlobal,
      lowerBoundGlobal,
      upperBoundGlobal)

  /** Construct a SubtractiveClustering model with default parameters. */
  def apply(numPartitions: Int): SubtractiveClustering =
    new SubtractiveClustering(numPartitions)
}
