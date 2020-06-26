/**
 * ModelTest.scala
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

import fuzzyspark.clustering.{ModelIdentification, SubtractiveClustering}
import fuzzyspark.frbs.{WM, WMModel}

import java.io.File
import java.io.PrintWriter
import scala.io.Source
import scala.math.sqrt

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

object ModelTest {

  /** Configuration parameters. */
  val hdfs = true
  val numPartitions = 10
  val saveFile = false
  val outFile = "output/out_model.txt"

  /** Print a string `s` to a file `f`. */
  def printToFile(f: String, s: String) = {
    val pw = new PrintWriter(new File(f))
    try pw.write(s) finally pw.close()
  }

  /** Format an array of input-output vectors for printing. */
  def outputToString(o: Array[(Vector, Vector)]): String = {
    o.map { case (x, y) =>
      Array(x.toString, y.toString).mkString(",").filterNot ( "[]" contains _ )
    }.mkString("\n")
  }

  /** Test the Model Identification algorithm based on Chiu's clustering algorithm. */
  def testChiuModel(
    trainingData: RDD[Vector],
    testData: RDD[Vector],
    ra: Double) = {
    val chiu = SubtractiveClustering(numPartitions).setRadius(ra)
    val centers = chiu.chiuGlobal(trainingData)

    val output = ModelIdentification(centers, chiu.getRadius)
      .predictOutputRDD(testData)
      .collect()
      .toArray

    if (saveFile) {
      printToFile(outFile, outputToString(output))
      println("Chiu: Saved input-output pairs to file " + outFile)
    }
    else {
      println("Chiu: Input-output pairs:\n" + outputToString(output) + "\n")
    }
  }

  /** Get the minimum and maximum value in each dimension of a set of points. */
  def getRange(data: RDD[Vector]): Array[(Double, Double)] = {
    val dataMax = data.reduce { case (x1, x2) =>
      Vectors.dense(
        x1.toArray.zip(x2.toArray).map ( pair => pair._1 max pair._2 )
      )
    }.toArray

    val dataMin = data.reduce { case (x1, x2) =>
      Vectors.dense(
        x1.toArray.zip(x2.toArray).map ( pair => pair._1 min pair._2 )
      )
    }.toArray

    dataMin.zip(dataMax)
  }

  /** Test the Model Identification algorithm based on Wang & Mendel algorithm. */
  def testWMModel(
    trainingData: RDD[(Vector, Vector)],
    testData: RDD[Vector],
    numRegions: Array[Int]) = {

    val inputData = trainingData.keys.cache()
    val outputData = trainingData.values.cache()
    val dataRange = getRange(inputData) ++ getRange(outputData)

    val output = WM.train(trainingData, numRegions, dataRange)
      .predictRDD(testData)
      .collect()
      .toArray

    if (saveFile) {
      printToFile(outFile, outputToString(output))
      println("WM: Saved input-output pairs to file " + outFile)
    }
    else {
      println("WM: Input-output pairs:\n" + outputToString(output) + "\n")
    }
  }

  /** Model Identification examples with fuzzyspark. */
  def main(args: Array[String]) = {
    // Spark environment configuration
    val conf = new SparkConf().setAppName("ModelTest")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("/tmp")

    // Get input file
    var trainingFile = args.headOption.getOrElse {
      Console.err.println("No training file provided. Aborting...")
      sys.exit(1)
    }
    if (hdfs)
      trainingFile = "file://" + trainingFile

    // Load training data into RDD with specified number of partitions
    val inputDims = 1
    val trainingInput = sc.textFile(trainingFile, numPartitions)
    val trainingData = trainingInput.map { line =>
      val x = Vectors.dense(line.split(",").map ( _.toDouble ))
      (Vectors.dense(x.toArray.slice(0, inputDims)),
       Vectors.dense(x.toArray.slice(inputDims, x.size)))
    }.cache()
    val joinedTrainingData = trainingData.map { case (x, y) =>
      Vectors.dense(x.toArray ++ y.toArray)
    }

    // Test file
    val testFile = if (hdfs) "file://" + args(1) else args(1)
    val testInput = sc.textFile(testFile, numPartitions)
    val testData = testInput.map { line =>
      Vectors.dense(line.split(",").map ( _.toDouble ))
    }.cache()

    println()

    // Test model identification functions
    testChiuModel(joinedTrainingData, testData, 0.3)
    testWMModel(trainingData, testData, Array(3, 3))

    // Stop spark
    sc.stop()
  }
}
