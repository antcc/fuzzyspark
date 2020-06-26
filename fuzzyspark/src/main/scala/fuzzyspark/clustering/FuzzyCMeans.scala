/**
 * FuzzyCMeans.scala
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
import scala.util.Random
import scala.math.{pow, sqrt}

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

/**
 * Fuzzy C Means clustering algorithm [1].
 *
 * [1] Bezdek, J. C. (2013). Pattern recognition with fuzzy objective
 *     function algorithms. Springer Science & Business Media.
 */
private class FuzzyCMeans(
  private var initMode: String,
  private var numPartitions: Int,
  private var c: Int,
  private var alpha:Double,
  private var initCenters: Option[Array[Vector]],
  private var chiuInstance: Option[SubtractiveClustering],
  private var m: Int,
  private var epsilon: Double,
  private var maxIter: Int,
  private var seed: Long
) extends Serializable {

  import FuzzyCMeans._
  import VectorUtils._

  /**
   * Constructs a FuzzyCMeans instance with default parameters:
   *   { c = 3, alpha = 0.5, initMode = RANDOM,
   *     initCenters = None, chiuInstance = None,
   *     m = 2, epsilon = 1e-3, maxIter = 100, seed = random }
   */
  def this(numPartitions: Int) =
    this(FuzzyCMeans.RANDOM, numPartitions, 3, 0.5, None, None, 2, 1e-3, 100, Random.nextLong)

  /** Initialization mode. */
  def getInitMode: String = initMode

  /**
   * Set initialization mode. Supported initialization modes are:
   *   - RANDOM: create `c` random centers.
   *   - PROVIDED: use centers in `initCenters`.
   *   - Chiu: apply subtractive clustering to create centers.
   *     There are three strategies, according to how the data is
   *     distributed across nodes:
   *     - CHIU_GLOBAL.
   *     - CHIU_LOCAL.
   *     - CHIU_INTERMEDIATE.
   *
   * @see [[fuzzyspark.clustering.SubtractiveClustering]]
   */
  def setInitMode(initMode: String): this.type = {
    require(
      validateInitMode(initMode),
      s"Initialization mode must be one of the supported values but got ${initMode}.")
    this.initMode = initMode
    this
  }

  /** Desired level of parallelism. */
  def getNumPartitions: Int = numPartitions

  /** Set desired level of parallelism. */
  def setNumPartitions(numPartitions: Int): this.type = {
    require(
      numPartitions >= 1,
      s"Number of partitions must be positive but got ${c}.")
    this.numPartitions = numPartitions
    this
  }

  /** Number of clusters to create. Takes the value 0 if it's yet undefined. */
  def getC: Int = c

  /**
   * Set number of clusters.
   *
   * @note Only relevant when initialization mode is RANDOM.
   */
  def setC(c: Int): this.type = {
    require(
      c >= 0,
      s"Number of clusters must be nonnegative but got ${c}.")
    this.c = c
    this
  }

  /** Threshold for alpha-cut. */
  def getAlpha: Double = alpha

  /** Set threshold for alpha-cut in classification. */
  def setAlpha(alpha: Double): this.type = {
    require(
      alpha >= 0 && alpha <= 1,
      s"Alpha must be in [0,1] but got ${alpha}.")
    this.alpha = alpha
    this
  }

  /** Initial centers. */
  def getInitCenters: Option[Array[Vector]] = initCenters

  /**
   * Set initial centers.
   *
   * @note Only relevant when initialization mode is PROVIDED.
   */
  def setInitCenters(initCenters: Option[Array[Vector]]): this.type = {
    this.initCenters = initCenters
    this
  }

  /** Instance of SubtractiveClustering for getting initial centers. */
  def getChiuInstance: Option[SubtractiveClustering] = chiuInstance

  /**
   * Set instance of SubtractiveClustering.
   *
   * @note Only relevant when initialization mode is CHIU_*.
   */
  def setChiuInstance(chiuInstance: Option[SubtractiveClustering]): this.type = {
    this.chiuInstance = chiuInstance
    this
  }

  /** Fuzziness degree. */
  def getM: Int = m

  /** Set fuzziness degree. */
  def setM(m: Int): this.type = {
    require(
      m >= 1,
      s"Fuzziness degree must be greater than one but got ${m}.")
    this.m = m
    this
  }

  /** Tolerance for stopping condition. */
  def getEpsilon: Double = epsilon

  /** Set tolerance for stopping condition. */
  def setEpsilon(epsilon: Double): this.type = {
    require(
      epsilon > 0,
      s"Tolerance must be positive but got ${epsilon}.")
    this.epsilon = epsilon
    this
  }

  /** Maximum number of iterations for stopping condition. */
  def getMaxIter: Int = maxIter

  /** Set maximum number of iterations for stopping conditions. */
  def setMaxIter(maxIter: Int): this.type = {
    require(
      maxIter > 0,
      s"Maximum number of iterations must be positive but got ${maxIter}.")
    this.maxIter = maxIter
    this
  }

  /** Seed for randomness. */
  def getSeed: Long = seed

  /** Set seed for randomness. */
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /**
   * Randomly initialize `c` distinct cluster centers.
   *
   * @note It's possible that fewer than `c` centers are
   * returned, if the data has less than `c` distinct points.
   */
  private def initRandom(data: RDD[Vector]) = {
    data.takeSample(false, c, seed).distinct
  }

  /**
   * Train a Fuzzy C Means model on the given set of points.
   * If the data is pre-labeled, it creates a model that can
   * be used for classification tasks.
   *
   * @note The RDD `labeledData` should be cached for high performance,
   * since this is an iterative algorithm.
   */
  def run(
    labeledData: RDD[(Vector, Double)],
    classification: Boolean): FuzzyCMeansModel = {
    val data = labeledData.keys.cache()
    val sc = data.sparkContext

    // Compute initial centers
    var centers: Array[Vector] = initMode match {
      case RANDOM =>
        initRandom(data)
      case PROVIDED =>
        initCenters.getOrElse(Array[Vector]())
      case CHIU_GLOBAL =>
        val chiuModel = chiuInstance
          .getOrElse(SubtractiveClustering(sqrt(numPartitions).toInt))
        chiuModel.chiuGlobal(data)
      case CHIU_LOCAL =>
        val chiuModel = chiuInstance
          .getOrElse(SubtractiveClustering(numPartitions))
        data.mapPartitionsWithIndex ( chiuModel.chiuLocal )
          .map ( _._2 )
          .collect()
          .toArray
      case CHIU_INTERMEDIATE =>
        val chiuModel = chiuInstance
          .getOrElse(SubtractiveClustering(numPartitions))
        chiuModel.chiuIntermediate(data)
    }
    c = centers.length
    require(c > 0, s"Number of centers must be positive but got ${c}.")

    // Compute initial membership matrix and loss
    var centersBroadcast = sc.broadcast(centers)
    var u = labeledData.map { case (x, l) =>
      ((x, l), computeRow(x, centersBroadcast.value, m))
    }
    if (numPartitions > labeledData.getNumPartitions) {
      u = u.repartition(numPartitions).cache()
    }
    else if (numPartitions < labeledData.getNumPartitions) {
      u = u.coalesce(numPartitions).cache()
    }
    else {
      u = u.cache()
    }
    var uOld = u

    // Main loop of the algorithm
    var iter = 0
    do {
      // Update centers
      var centersWithIndex = centers.zipWithIndex
      for ((c, j) <- centersWithIndex) {
        var sums = u.map { case ((x, _), r) =>
          (multiply(x, pow(r(j), m)), pow(r(j), m))
        }.reduce { case (t, s) =>
          (add(t._1, s._1), t._2 + s._2)
        }

        centers(j) = divide(sums._1, sums._2)
      }
      centersBroadcast = sc.broadcast(centers)

      // Update membership matrix
      uOld = u
      u = uOld.map { case ((x, l), _) =>
        ((x, l), computeRow(x, centersBroadcast.value, m))
      }.cache()

      if (iter % 30 == 0) {
        u.checkpoint()
        u.count()
      }

      // Increase iteration count
      iter += 1
      if (iter % 10 == 0)
        println("[FCM] Iteration #" + iter)

    } while (iter < maxIter && !stoppingCondition(uOld, u))

    var labelsCenters: Option[Array[Double]] = None
    if (classification)
      labelsCenters = Option(computeCentersLabels(centersBroadcast.value, u))

    FuzzyCMeansModel(
      centers, m,
      computeLoss(u, centersBroadcast.value, m),
      iter, labelsCenters)
  }

  /**
   * Check if the algorithm should stop according to the
   * stopping condition.
   *
   * The algorithm is considered to have converged if the
   * shift in any position of the membership matrix is
   * no greater than `epsilon`.
   */
  private def stoppingCondition(
    old: RDD[((Vector, Double), Vector)],
    curr: RDD[((Vector, Double), Vector)]): Boolean = {
    val diff = old.join(curr).map { case ((_, _), (u, v)) =>
        val t = abs(subtract(v, u))
        t(t.argmax)
    }.max()

    diff < epsilon
  }

  /**
   * Assign a label to each cluster center.
   *
   * For a particular center point, we consider the membership
   * of each data point to it and perform an alpha-cut at 0.6.
   * Of the remaining points, we compute the predominant class,
   * and thus assign that label to the center in question.
   *
   * @param centers Cluster centers found
   * @param u Membership matrix associated with the labeled data points a
   * and the centers
   */
  private def computeCentersLabels(
    centers: Array[Vector],
    u: RDD[((Vector, Double), Vector)]): Array[Double] = {

    // Count relevant labels for each cluster center
    val labelsCount = u.flatMap { case ((x, l), r) =>
      r.toArray.zipWithIndex.map { case (mu, i) =>
        ((l, i), if (mu > alpha) 1 else 0)
      }
    }.reduceByKey ( _ + _ )

    // Choose the predominant class for each center
    val c = centers.size
    val labels = Array.ofDim[Double](c)
    for (i <- 0 until c) {
      labels(i) = labelsCount.filter ( _._1._2 == i )
        .max()(Ordering[Int].on ( _._2 ))._1._1
    }

    labels
  }

  /**
   * Validate initialization mode.
   *
   * @see [[fuzzyspark.clustering.FuzzyCMeans.setInitMode]]
   */
  private def validateInitMode(initMode: String) = initMode match {
    case RANDOM => true
    case CHIU_GLOBAL => true
    case CHIU_LOCAL => true
    case CHIU_INTERMEDIATE => true
    case PROVIDED => true
    case _ => false
  }
}

/** Top-level methods for calling Fuzzy C Means clustering. */
object FuzzyCMeans {

  /** Initialization mode names */
  val RANDOM = "Random"
  val CHIU_GLOBAL = "Chiu Global"
  val CHIU_LOCAL = "Chiu Local"
  val CHIU_INTERMEDIATE = "Chiu Intermediate"
  val PROVIDED = "Provided"

  /**
   * Train a Fuzzy C Means model using specified parameters.
   *
   * @param labeledData Labeled training points.
   * @param initMode The initialization algorithm.
   * @param numPartitions Desired number of partitions.
   * @param c Number of clusters to create.
   * @param alpha Threshold for alpha-cut in classification
   * @param initCenters Initial cluster centers.
   * @param chiuInstance Instance of SubtractiveClustering.
   * @param m Fuzziness degree.
   * @param epsilon Tolerance for stopping condition.
   * @param maxIter Maximum number of iterations allowed.
   * @param seed Seed for randomness.
   * @param classification Whether to prepare labels for classification.
   */
  def train(
    labeledData: RDD[(Vector, Double)],
    initMode: String,
    numPartitions: Int,
    c: Int = 0,
    alpha: Double = 0.5,
    initCenters: Option[Array[Vector]] = None,
    chiuInstance: Option[SubtractiveClustering] = None,
    m: Int = 2,
    epsilon: Double = 1e-3,
    maxIter: Int = 100,
    seed: Long = Random.nextLong,
    classification: Boolean = false): FuzzyCMeansModel = {
    new FuzzyCMeans(
      initMode,
      numPartitions,
      c,
      alpha,
      initCenters,
      chiuInstance,
      m,
      epsilon,
      maxIter,
      seed).run(labeledData, classification)
  }

  /**
   * Compute a row of the membership matrix pertaining
   * to a point `x` and the specified cluster centers,
   * with fuzziness degree `m`.
   *
   * The membership matrix is a key-value RDD indexed by the
   * data points. In every row the membership of the corresponding
   * point to every cluster center is computed.
   */
  private[clustering] def computeRow(
    x: Vector,
    centers: Array[Vector],
    m: Int) = {
    val c = centers.length
    val membership = Array.ofDim[Double](c)

    for (j <- 0 until c) {
      val denom = centers.map { ck =>
        pow(
          sqrt(Vectors.sqdist(x, centers(j))) / sqrt(Vectors.sqdist(x, ck)),
          2.0 / (m - 1))
      }.sum

      if (denom.isInfinity)
        membership(j) = 0.0
      else if (denom.isNaN)
        membership(j) = 1.0
      else
        membership(j) = 1.0 / denom
    }

    Vectors.dense(membership)
  }

  /**
   * Compute loss function for given membership matrix and centers,
   * with fuzziness degree `m`.
   */
  private[clustering] def computeLoss(
    u: RDD[((Vector, Double), Vector)],
    centers: Array[Vector],
    m: Int) = {
    u.map { case ((x, _), r) =>
      r.toArray.zip(centers).map { case (e, c) =>
        pow(e, m) * Vectors.sqdist(x, c)
      }.sum
    }.sum
  }
}
