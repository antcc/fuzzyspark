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
  private var m: Double,
  private var epsilon: Double,
  private var maxIter: Int,
  private var matrixNorm: String,
  private var scaleAlpha: Boolean,
  private var seed: Long
) extends Serializable {

  import FuzzyCMeans._
  import VectorUtils._

  /**
   * Constructs a FuzzyCMeans instance with default parameters:
   *   { c = 3, alpha = 0.5, initMode = RANDOM,
   *     initCenters = None, chiuInstance = None,
   *     m = 2.0, epsilon = 1e-4, maxIter = 100,
   *     matrixNorm = MAX, scaleAlpha = true, seed = random }
   */
  def this(numPartitions: Int) =
    this(
      FuzzyCMeans.RANDOM, numPartitions,
      3, 0.5, None, None, 2.0, 1e-4, 100,
      FuzzyCMeans.MAX, true, Random.nextLong)

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
  def getM: Double = m

  /** Set fuzziness degree. */
  def setM(m: Double): this.type = {
    require(
      m > 1,
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

  /** Matrix norm used in stopping condition. */
  def getMatrixNorm: String = matrixNorm

  /**
   * Set matrix norm. Supported values are:
   *   - FROBENIUS: Frobenius Norm.
   *   - MAX: Max Norm (infinity norm).
   */
  def setMatrixNorm(matrixNorm: String): this.type = {
    require(
      validateMatrixNorm(matrixNorm),
      s"Matrix norm must be one of the supported values but got ${matrixNorm}.")
    this.matrixNorm = matrixNorm
    this
  }

  /** Whether to scale the value of alpha for the alpha-cut. */
  def getScaleAlpha: Boolean = scaleAlpha

  /** Set scaleAlpha value. */
  def setAlpha(alpha: Boolean): this.type = {
    this.scaleAlpha = scaleAlpha
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
   *
   * @note The RDD `trainingData` should be cached for high performance,
   * since this is an iterative algorithm.
   */
  def run(trainingData: RDD[Vector]): FuzzyCMeansModel = {
    val sc = trainingData.sparkContext

    // Compute initial centers
    var centers: Array[Vector] = initMode match {
      case RANDOM =>
        initRandom(trainingData)
      case PROVIDED =>
        initCenters.getOrElse(Array[Vector]())
      case CHIU_GLOBAL =>
        val chiuModel = chiuInstance
          .getOrElse(SubtractiveClustering(sqrt(numPartitions).toInt))
        chiuModel.chiuGlobal(trainingData)
      case CHIU_LOCAL =>
        val chiuModel = chiuInstance
          .getOrElse(SubtractiveClustering(numPartitions))
        trainingData.mapPartitionsWithIndex ( chiuModel.chiuLocal )
          .map ( _._2 )
          .collect()
          .toArray
      case CHIU_INTERMEDIATE =>
        val chiuModel = chiuInstance
          .getOrElse(SubtractiveClustering(numPartitions))
        chiuModel.chiuIntermediate(trainingData)
    }
    c = centers.length
    require(c > 0, s"Number of centers must be positive but got ${c}.")
    var centersNew = Array.ofDim[Vector](c)

    // Repartition data if necessary
    var data = trainingData
    if (numPartitions > data.getNumPartitions) {
      data = data.repartition(numPartitions).cache()
    }
    else if (numPartitions < data.getNumPartitions) {
      data = data.coalesce(numPartitions).cache()
    }
    // Main loop of the algorithm
    var iter = 0
    var stop = false
    while (iter < maxIter && !stop) {
      // Broadcast centers to all nodes
      var centersBc = sc.broadcast(centers)

      // Calculate sum of [u_ij]^m and [u_ij]^m * x
      var membershipMatrixSum = data.mapPartitions { iterator =>
        for {
          x <- iterator
          j <- 0 until c
        } yield {
          val denom = centersBc.value.map { ck =>
            pow(sqrt(Vectors.sqdist(x, ck)), -2.0 / (m - 1))
          }.sum
          var membership =
            pow(sqrt(Vectors.sqdist(x, centersBc.value(j))), -2.0 / (m - 1)) / denom

          if (membership.isNaN)
            membership = 1.0

          (j, (multiply(x, pow(membership, m)), pow(membership, m)))
        }
      }.reduceByKey { case ((dx1, d1), (dx2, d2)) =>
        (add(dx1, dx2), d1 + d2)
      }.collect

      // Update centers
      for ((j, (dx, d)) <- membershipMatrixSum)
        centersNew(j) = divide(dx, d)

      if (stoppingCondition(centers, centersNew))
        stop = true
      else
        centers = centersNew.clone()

      // Increase iteration count
      iter += 1
    }

    FuzzyCMeansModel(
      centersNew, m,
      iter, None)
  }

  /**
   * Train a Fuzzy C Means model on pre-labeled data, creating a model
   * that can be used for classification tasks.
   */
  def runClassifier(labeledData: RDD[(Vector, Double)]): FuzzyCMeansModel = {
    // Train model
    val data = labeledData.keys.cache()
    val model = run(data)

    // Set labels
    val labels = computeCentersLabels(
      labeledData, model.clusterCenters,
      m, alpha, scaleAlpha)
    model.labels = Option(labels)

    model
  }

  /**
   * Check if the algorithm should stop according to the
   * stopping condition.
   *
   * The algorithm is considered to have converged if the
   * considered matrix norm of the shift in the centers is no
   * greater than `epsilon`.
   */
  private def stoppingCondition(
      centers: Array[Vector],
      centersNew: Array[Vector]): Boolean = {
    val diff = matrixNorm match {
      case FROBENIUS =>
        sqrt(centers.zip(centersNew).map { case (c, cNew) =>
          Vectors.sqdist(c, cNew)
        }.sum)
      case MAX =>
        centers.zip(centersNew).map { case (c, cNew) =>
            abs(subtract(cNew, c)).toArray.sum
        }.max
    }

    diff < epsilon
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

  /**
   * Validate matrix norm type.
   *
   * @see [[fuzzyspark.clustering.FuzzyCMeans.setMatrixNorm]]
   */
  private def validateMatrixNorm(matrixNorm: String) = matrixNorm match {
    case FROBENIUS => true
    case MAX => true
    case _ => false
  }
}

/** Top-level methods for calling Fuzzy C Means clustering. */
object FuzzyCMeans {

  /** Initialization mode names. */
  val RANDOM = "Random"
  val CHIU_GLOBAL = "Chiu Global"
  val CHIU_LOCAL = "Chiu Local"
  val CHIU_INTERMEDIATE = "Chiu Intermediate"
  val PROVIDED = "Provided"

  /** Matrix norm names. */
  val FROBENIUS = "Frobenius Norm"
  val MAX = "Max Norm"

  /**
   * Train a Fuzzy C Means model using specified parameters.
   *
   * @param trainingData Training data points.
   * @param initMode The initialization algorithm.
   * @param numPartitions Desired number of partitions.
   * @param c Number of clusters to create.
   * @param alpha Threshold for alpha-cut in classification
   * @param initCenters Initial cluster centers.
   * @param chiuInstance Instance of SubtractiveClustering.
   * @param m Fuzziness degree.
   * @param epsilon Tolerance for stopping condition.
   * @param maxIter Maximum number of iterations allowed.
   * @param matrixNorm Matrix norm used for stopping condition.
   * @param scaleAlpha Whether to scale the value of the threshold for alpha cut.
   * @param seed Seed for randomness.
   */
  def fit(
      trainingData: RDD[Vector],
      initMode: String,
      numPartitions: Int,
      c: Int = 0,
      alpha: Double = 0.5,
      initCenters: Option[Array[Vector]] = None,
      chiuInstance: Option[SubtractiveClustering] = None,
      m: Double = 2.0,
      epsilon: Double = 1e-4,
      maxIter: Int = 100,
      matrixNorm: String = MAX,
      scaleAlpha: Boolean = true,
      seed: Long = Random.nextLong): FuzzyCMeansModel = {
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
      matrixNorm,
      scaleAlpha,
      seed).run(trainingData)
  }

  /**
   * Train a Fuzzy C Means model using specified parameters,
   * for a classification problem.
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
   * @param matrixNorm Matrix norm used for stopping condition.
   * @param scaleAlpha Whether to scale the value of the threshold for alpha cut.
   * @param seed Seed for randomness.
   */
  def fitClassifier(
      labeledData: RDD[(Vector, Double)],
      initMode: String,
      numPartitions: Int,
      c: Int = 0,
      alpha: Double = 0.5,
      initCenters: Option[Array[Vector]] = None,
      chiuInstance: Option[SubtractiveClustering] = None,
      m: Double = 2.0,
      epsilon: Double = 1e-4,
      maxIter: Int = 100,
      matrixNorm: String = MAX,
      scaleAlpha: Boolean = true,
      seed: Long = Random.nextLong): FuzzyCMeansModel = {
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
      matrixNorm,
      scaleAlpha,
      seed).runClassifier(labeledData)
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
  private[clustering] def computeMembershipRow(
      x: Vector,
      centers: Array[Vector],
      m: Double) = {
    val c = centers.length
    val membership = Array.ofDim[Double](c)

    for (j <- 0 until c) {
      val denom = centers.map { ck =>
        pow(sqrt(Vectors.sqdist(x, ck)), -2.0 / (m - 1))
      }.sum
      var mu = pow(sqrt(Vectors.sqdist(x, centers(j))), -2.0 / (m - 1)) / denom

      if (mu.isNaN)
        mu = 1.0

      membership(j) = mu
    }

    membership
  }

  /**
   * Assign a label to each cluster center.
   *
   * For a particular center point, we consider the membership
   * of each data point to it and perform an alpha-cut at a
   * predefined value. Of the remaining points, we compute the
   * predominant class, and thus assign that label to the center in question.
   *
   * @param labeledData Training data points along with their labels
   * @param centers Cluster centers found
   */
  private[clustering] def computeCentersLabels(
      labeledData: RDD[(Vector, Double)],
      centers: Array[Vector],
      m: Double,
      alpha: Double,
      scaleAlpha: Boolean = true): Array[Double] = {
    val sc = labeledData.sparkContext
    val centersBc = sc.broadcast(centers)
    val c = centers.size

    var maxMembership = if (scaleAlpha) {
      // Get max membership degree to every cluster center
      labeledData.map { case (x, l) =>
        computeMembershipRow(x, centersBc.value, m)
      }.reduce { case (r1, r2) =>
        r1.zip(r2).map { case (c1, c2) => c1 max c2 }
      }
    } else {
      // Assume max membership is 1.0 for every cluster
      Array.fill(c)(1.0)
    }

    // Count relevant labels for each cluster center
    val maxMembershipBc = sc.broadcast(maxMembership)
    val labelsCount = labeledData.flatMap { case (x, l) =>
      val r = computeMembershipRow(x, centersBc.value, m)
      r.zipWithIndex.map { case (mu, j) =>
        ((l, j), if (mu > alpha * maxMembershipBc.value(j)) 1 else 0)
      }
    }.reduceByKey ( _ + _ )

    // Choose the predominant class for each center
    val labels = Array.ofDim[Double](c)
    for (j <- 0 until c) {
      labels(j) = labelsCount.filter ( _._1._2 == j )
        .max()(Ordering[Int].on ( _._2 ))._1._1
    }

    labels
  }

  /**
   * Compute loss function between cluster centers and data points.
   */
  private[clustering] def computeLoss(
      data: RDD[Vector],
      centers: Array[Vector],
      m: Double) = {
    val sc = data.sparkContext
    val centersBc = sc.broadcast(centers)

    val loss = data.map { x =>
      val r = computeMembershipRow(x, centersBc.value, m)
      r.zip(centersBc.value).map { case (mu, c) =>
        pow(mu, m) * Vectors.sqdist(x, c)
      }.sum
    }.sum

    loss
  }
}
