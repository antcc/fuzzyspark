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
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

object ClusteringTest {

  /** Configuration parameters. */
  val hdfs = true
  val numPartitions = 37 * 5
  val numPartitionsPerGroup = 10
  val saveFile = true
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
      for (i <- 0 until c.size) {
        result += c(i)
        if (i < c.size - 1)
          result += ","
        else
          result += "\n"
      }
    }
    result
  }

  /** Test the Subtractive Clustering global algorithm. */
  def testChiuGlobal(data: RDD[Vector]) = {
    val chiu = SubtractiveClustering(numPartitions)
    val centers = chiu.chiuGlobal(data)
    println("\n\n--> NO. OF CENTERS: " + centers.length + "\n\n")
  }

  /** Test the Subtractive Clustering local algorithm. */
  def testChiuLocal(data: RDD[Vector]) = {
    val chiu = SubtractiveClustering(numPartitions)
    val centers = data.mapPartitionsWithIndex ( chiu.chiuLocal )
      .map ( _._2 )
      .collect()
      .toArray
    println("\n\n--> NO. OF CENTERS: " + centers.length + "\n\n")
  }

  /** Test the Subtractive Clustering intermediate algorithm. */
  def testChiuIntermediate(data: RDD[Vector]) = {
    val chiu = SubtractiveClustering(0.3, 0.15, 0.5, numPartitions, numPartitionsPerGroup)
    val centers = chiu.chiuIntermediate(data)
    println("\n\n--> NO. OF CENTERS: " + centers.length + "\n\n")
  }

  /** Train a Fuzzy C Means model. */
  def testFuzzyCMeans(data: RDD[Vector]) = {
    val fcmModel = FuzzyCMeans.train(
      data,
      initMode = FuzzyCMeans.CHIU_INTERMEDIATE,
      numPartitions = 37 * 3,
      chiuInstance =
        Option(SubtractiveClustering(0.3, 0.15, 0.5, numPartitions, numPartitionsPerGroup))
    )

    println("\n\n--> NO. OF CENTERS: " + fcmModel.c)
    println("--> LOSS: " + fcmModel.trainingLoss)
    println("--> NO. OF ITERATIONS: " + fcmModel.trainingIter + "\n")
    if (saveFile) {
      printToFile(outFile, centersToString(fcmModel.clusterCenters))
      println("--> SAVED CENTERS TO FILE " + outFile + "\n\n")
    }
    else {
      println("--> CLUSTER CENTERS:\n" +
        fcmModel.clusterCenters.map ( _.toString ).mkString("\n")) +
        "\n\n"
    }
  }

  /** Clustering examples with fuzzyspark. */
  def main(args: Array[String]) = {
    // Spark environment configuration
    val conf = new SparkConf().setAppName("ClusteringTest")
    val sc = new SparkContext(conf)

    // Get input file
    var inputFile = args.headOption.getOrElse {
      Console.err.println("\n\n--> NO INPUT FILE PROVIDED. ABORTING...\n")
      sys.exit(1)
    }
    if (hdfs)
      inputFile = "file://" + inputFile

    // Load data into RDD with specified number of partitions
    val input = sc.textFile(inputFile, numPartitions)
    val data = input.map { line =>
        Vectors.dense(line.split(",").map ( _.toDouble ))
    }.cache()

    // Test functions
    //testChiuGlobal(data)
    //testChiuLocal(data)
    //testChiuIntermediate(data)
    testFuzzyCMeans(data)

    // Stop spark
    sc.stop()
  }
}
