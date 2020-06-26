/**
 * Utils.scala
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

package fuzzyspark

/** Utility methods for basic operations with Vector type. */
private[fuzzyspark] object VectorUtils {

  import breeze.linalg.{DenseVector => BreezeVector}
  import scala.math.{abs => fabs}

  import org.apache.spark.mllib.linalg.{Vector, Vectors}

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
