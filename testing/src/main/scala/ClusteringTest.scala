/**
 * ClusteringTest.scala
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

import fuzzyspark.clustering.{FuzzyCMeans, SubtractiveClustering}

import java.io.File
import java.io.PrintWriter
import scala.io.Source

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.{SparkContext, SparkConf}

object ClusteringTest {

  /** Configuration parameters. */
  val hdfs = false
  val numPartitions = 4
  val saveFile = false
  val outFile = "output/centers_norm.txt"

  /** Print a string `s` to a file `f`. */
  def printToFile(f: String, s: String) = {
    val pw = new PrintWriter(new File(f))
    try pw.write(s) finally pw.close()
  }

  /** Format an array of vectors for printing. */
  def centersToString(centers: Array[Vector]): String = {
    var result = ""
    for (c <- centers) {
      for (i <- 0 to c.size - 1) {
        result += c(i)
        if (i != c.size - 1)
          result += ","
        else
          result += "\n"
      }
    }
    result
  }

  /** Clustering example with Fuzzy C Means. */
  def main(args: Array[String]) = {
    // Spark environment configuration
    val conf = new SparkConf().setAppName("FCMClusteringTest")
    val sc = new SparkContext(conf)

    // Get input file
    var inputFile = args.headOption.getOrElse {
      Console.err.println("No input file provided. Aborting...")
      sys.exit(1)
    }
    if (hdfs)
      inputFile = "file://" + inputFile

    // Load data into RDD with specified number of partitions
    val input = sc.textFile(inputFile, numPartitions)
    val data = input.map { line =>
        Vectors.dense(line.split(",").map ( _.toDouble ))
    }.cache()

    // Test Chiu Global
    /**val chiu = SubtractiveClustering(0.3, 0.15, 0.5, numPartitions)
    val centers = chiu.chiuGlobal(data)
    println("\n\n--> NO. OF CENTERS: " + centers.length + "\n\n")*/

    // Test Chiu Local
    val centers2 = data.mapPartitionsWithIndex ( chiu.chiuLocal )
      .map ( _._2 )
      .collect()
      .toArray
    println("\n\n--> NO. OF CENTERS: " + centers.length + "\n\n")

    // Train a FCM model on the data using initial centers from Chiu
    /**val fcmModel = FuzzyCMeans.train(
      data,
      initMode = FuzzyCMeans.CHIU_GLOBAL,
      chiuInstance = Option(SubtractiveClustering(0.3, 0.15, 0.5, numPartitions))
    )

    // Print cluster centers
    println("\n--> NO. OF CENTERS: " + fcmModel.c)
    println("--> LOSS: " + fcmModel.trainingLoss)
    println("--> NO. OF ITERATIONS: " + fcmModel.trainingIter)
    if (saveFile)
      printToFile(outFile, centersToString(fcmModel.clusterCenters))
    else
      println("--> CLUSTER CENTERS:\n" +
        fcmModel.clusterCenters.map ( _.toString ).mkString("\n"))*/

    // Stop spark
    sc.stop()
  }
}
