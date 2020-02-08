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

import scala.util.Random
import scala.math.{abs => fabs, pow, sqrt}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

/** Fuzzy C Means clustering algorithm. */
private class FuzzyCMeans(
  private var initMode: String,
  private var c: Option[Int],
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
   *   { c = 3, initMode = RANDOM, initCenters = None,
   *     chiuInstance = None, m = 2, epsilon = 1e-6,
   *     maxIter = 100, seed = random }
   */
  def this() =
    this(FuzzyCMeans.RANDOM, Option(3), None, None, 2, 1e-6, 100, Random.nextLong)

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

  /** Number of clusters to create. */
  def getC: Option[Int] = c

  /**
   * Set number of clusters.
   *
   * @note Only relevant when initialization mode is RANDOM.
   */
  def setC(c: Option[Int]): this.type = {
    if (c.isDefined) {
      require(
        c.get > 0,
        s"Number of clusters must be positive but got ${c}.")
    }
    this.c = c
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
    data.takeSample(false, c.getOrElse(0), seed).distinct
  }

  /**
   * Train a Fuzzy C Means model on the given set of points.
   *
   * @note The RDD `data` should be cached for high performance,
   * since this is an iterative algorithm.
   */
  def run(data: RDD[Vector]): FuzzyCMeansModel = {
    val sc = data.sparkContext

    // Compute initial centers
    val chiuModel = chiuInstance.getOrElse(SubtractiveClustering())
    var centers: Array[Vector] = initMode match {
      case RANDOM =>
        initRandom(data)
      case CHIU_GLOBAL =>
        chiuModel.chiuGlobal(data)
      case CHIU_LOCAL =>
        data.mapPartitionsWithIndex ( chiuModel.chiuLocal )
          .map ( _._2 )
          .collect()
          .toArray
      case CHIU_INTERMEDIATE =>
        data.mapPartitionsWithIndex ( chiuModel.chiuIntermediate )
          .collect()
          .toArray
      case PROVIDED =>
        initCenters.getOrElse(Array[Vector]())
    }
    c = Option(centers.length)
    require(c.get > 0, s"Number of centers must be positive but got ${c.get}.")

    // Compute initial membership matrix and loss
    var centersBroadcast = sc.broadcast(centers)
    var u = data.map ( computeRow(_, centersBroadcast, m) ).cache()
    var uOld = u

    // Main loop of the algorithm
    var iter = 0
    do {
      // Update centers
      var centersWithIndex = centers.zipWithIndex
      for ((c, j) <- centersWithIndex) {
        var sums = u.map { case (x, r) =>
          (multiply(x, pow(r(j), m)), pow(r(j), m))
        }.reduce { case (t, s) =>
          (add(t._1, s._1), t._2 + s._2)
        }

        centers(j) = divide(sums._1, sums._2)
      }
      centersBroadcast = sc.broadcast(centers)

      // Update membership matrix
      uOld = u
      u = uOld.map { case (x, r) => computeRow(x, centersBroadcast, m) }.cache()

      // Increase iteration count
      iter += 1

    } while (iter < maxIter && !stoppingCondition(uOld, u))

    FuzzyCMeansModel(centers, m, computeLoss(u, centersBroadcast, m), iter)
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
    old: RDD[(Vector, Vector)],
    curr: RDD[(Vector, Vector)]): Boolean = {
    val diff = old.join(curr).map { case (x, (u, v)) =>
        val t = abs(subtract(v, u))
        t(t.argmax)
    }.max()

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
   * @param data Training points as an `RDD` of `Vector` types.
   * @param initMode The initialization algorithm.
   * @param c Number of clusters to create.
   * @param initCenters Initial cluster centers.
   * @param chiuInstance Instance of SubtractiveClustering.
   * @param m Fuzziness degree.
   * @param epsilon Tolerance for stopping condition.
   * @param maxIter Maximum number of iterations allowed.
   * @param seed Seed for randomness.
   */
  def train(
    data: RDD[Vector],
    initMode: String,
    c: Option[Int],
    initCenters: Option[Array[Vector]],
    chiuInstance: Option[SubtractiveClustering],
    m: Int,
    epsilon: Double,
    maxIter: Int,
    seed: Long): FuzzyCMeansModel = {
    new FuzzyCMeans(
      initMode,
      c,
      initCenters,
      chiuInstance,
      m,
      epsilon,
      maxIter,
      seed).run(data)
  }

  /**
   * Train a Fuzzy C Means model using specified parameters,
   * and default parameters for the rest.
   *
   * @param data Training points as an `RDD` of `Vector` types.
   * @param initMode The initialization algorithm.
   * @param c Number of clusters to create.
   * @param initCenters Initial cluster centers.
   */
  def train(
    data: RDD[Vector],
    initMode: String,
    c: Option[Int] = None,
    initCenters: Option[Array[Vector]] = None,
    chiuInstance: Option[SubtractiveClustering] = None): FuzzyCMeansModel = {
    new FuzzyCMeans().setC(c)
      .setInitMode(initMode)
      .setChiuInstance(chiuInstance)
      .setInitCenters(initCenters)
      .run(data)
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
    centers: Broadcast[Array[Vector]],
    m: Int) = {
    val c = centers.value.length
    val membership = Array.ofDim[Double](c)

    for (j <- 0 until c) {
      val denom = centers.value.map { ck =>
        pow(
          sqrt(Vectors.sqdist(x, centers.value(j))) / sqrt(Vectors.sqdist(x, ck)),
          2.0 / (m - 1))
      }.sum

      if (denom.isInfinity)
        membership(j) = 0.0
      else if (denom.isNaN)
        membership(j) = 1.0
      else
        membership(j) = 1.0 / denom
    }

    (x, Vectors.dense(membership))
  }

  /**
   * Compute loss function for given membership matrix and centers,
   * with fuzziness degree `m`.
   */
  private[clustering] def computeLoss(
    u: RDD[(Vector, Vector)],
    centers: Broadcast[Array[Vector]],
    m: Int) = {
    u.map { case (x, r) =>
      r.toArray.zip(centers.value).map { case(e, c) =>
        pow(e, m) * Vectors.sqdist(x, c)
      }.sum
    }.sum
  }
}

/** Utility methods for basic operations with Vector type. */
private[clustering] object VectorUtils {

  import breeze.linalg.{DenseVector => BreezeVector}

  /** Element-wise sum of two vectors. */
  def add(x: Vector, y: Vector) = {
    var bx = new BreezeVector(x.toArray)
    var by = new BreezeVector(y.toArray)
    Vectors.dense((bx + by).toArray)
  }

  /** Element-wise subtraction of two vectors. */
  def subtract(x: Vector, y: Vector) = {
    var bx = new BreezeVector(x.toArray)
    var by = new BreezeVector(y.toArray)
    Vectors.dense((bx - by).toArray)
  }

  /** Product of every value of a vector with a scalar. */
  def multiply(x: Vector, l: Double) = {
    var bx = new BreezeVector(x.toArray)
    Vectors.dense((bx * l).toArray)
  }

  /** Division of every value of a vector with a scalar. */
  def divide(x: Vector, l: Double) = {
    require(
      l != 0,
      s"Scalar value to divide must be nonzero.")
    var bx = new BreezeVector(x.toArray)
    Vectors.dense((bx / l).toArray)
  }

  /** Element-wise absolute value of a vector. */
  def abs(x: Vector) = {
    Vectors.dense(x.toArray.map ( l => fabs(l) ))
  }
}
