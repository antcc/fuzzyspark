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
import scala.math.pow

import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.{SparkContext, SparkConf}

object ModelTest {

  /** Configuration parameters. */
  val hdfs = true
  val numPartitions = 20 * 19
  val numPartitionsPerGroup = 19 * 2
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
    testData: RDD[(Vector, Vector)],
    chiu: SubtractiveClustering,
    raModel: Double) = {
    val centers = chiu.chiuIntermediate(trainingData)
    println("--> Chiu number of centers found = " + centers.size)

    val model = ModelIdentification(centers, raModel)
    val testMSE =
      testData.map { case (y, z) =>
        pow(model.predictOutput(y)(0) - z(z.size-1), 2)
      }.sum / testData.count()
    println(f"--> Chiu Test MSE = $testMSE%1.3f%%")
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
    testData: RDD[(Vector, Vector)],
    numRegions: Array[Int]) = {

    val inputData = trainingData.keys.cache()
    val outputData = trainingData.values.cache()
    val dataRange = getRange(inputData) ++ getRange(outputData)

    val model = WM.train(trainingData, numRegions, dataRange)

    val testMSE =
      testData.map { case (y, z) =>
        pow(model.predict(y)(0) - z(z.size - 1), 2)
      }.sum / testData.count()
    println(f"--> WM Test MSE = $testMSE%1.3f%%")
  }

  /** Test Random Forest regression algorithm. */
  def testRandomForest(
    labeledTrainingData: RDD[(Vector, Vector)],
    testData: RDD[(Vector, Vector)],
    maxDepth: Int) = {
    val trainingData = labeledTrainingData.map { case (x, l) =>
      LabeledPoint(l(0), x)
    }.cache()
    val categoricalFeaturesInfo = Map[Int, Int]()  // All features are continuous.
    val numTrees = 200
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "variance"
    val maxBins = 100

    // Train RandomForest model
    val model = RandomForest.trainRegressor(
      trainingData, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, 42)

    // Evaluate model on test instances
    val labelAndPreds = testData.map { case (x, l) =>
      (l(0), model.predict(x))
    }

    // Compute regression error
    val testMSE = labelAndPreds.map{ case (v, p) => math.pow((v - p), 2) }.mean()
    println(f"--> RandomForest (# = 200) Test MSE = $testMSE%1.3f%%")
  }

  /** Test SVM classification algorithm. */
  def testLinearRegression(
    labeledTrainingData: RDD[(Vector, Vector)],
    testData: RDD[(Vector, Vector)]) = {
    val trainingData = labeledTrainingData.map { case (x, l) =>
      LabeledPoint(l(0), x)
    }.cache()

    // Run training algorithm to build the model
    val model = new RidgeRegressionWithSGD()
      .run(trainingData)

    // Evaluate model on test instances
    val labelAndPreds = testData.map { case (x, l) =>
      (l(0), model.predict(x))
    }

    // Compute regression error
    val testMSE = labelAndPreds.map{ case (v, p) => math.pow((v - p), 2) }.mean()
    println(f"--> RidgeRegression Test MSE = $testMSE%1.3f%%")
  }

  /** Measure execution time of a block. */
  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    val time = (t1 - t0) / (60.0 * 1e9)
    println(f"Total elapsed time: $time%1.2f min\n")
    result
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
    val trainingInput = sc.textFile(trainingFile, numPartitions)
    var joinedTrainingData = trainingInput.map { line =>
      Vectors.dense(line.split(",").map ( _.toDouble ))
    }.cache()

    var scaler = new StandardScaler(withMean = true, withStd = true).fit(joinedTrainingData)

    // Test file
    val testFile = if (hdfs) "file://" + args(1) else args(1)
    val testInput = sc.textFile(testFile, numPartitions)
    var joinedTestData = testInput.map { line =>
      Vectors.dense(line.split(",").map ( _.toDouble ))
    }.cache()

    // Scale train and test
    joinedTrainingData = joinedTrainingData.map { x => scaler.transform(x) }
    joinedTestData = joinedTestData.map { x => scaler.transform(x) }

    val inputDims = 6
    val trainingData = joinedTrainingData.map { x =>
      (Vectors.dense(x.toArray.slice(0, inputDims)),
       Vectors.dense(x.toArray.slice(inputDims, x.size)))
    }.cache()
    val testData = joinedTestData.map { x =>
      (Vectors.dense(x.toArray.slice(0, inputDims)),
       Vectors.dense(x.toArray.slice(inputDims, x.size)))
    }.cache()

    val ra = 1.0
    val rb = 1.5
    val raGlobal = 1.0
    val rbGlobal = 1.5
    val lb = 0.15
    val ub = 0.5
    val lbGlobal = 0.15
    val ubGlobal = 0.5
    val raModel = 1.0

    val chiu =
      SubtractiveClustering(
        ra, rb, lb, ub, numPartitions,
        numPartitionsPerGroup,
        raGlobal, rbGlobal, lbGlobal, ubGlobal)

    // Test model identification functions
    time { testRandomForest(trainingData, testData, 15) }
    time { testLinearRegression(trainingData, testData) }

    time { testChiuModel(joinedTrainingData, testData, chiu, raModel) }

    val params = Array(
      Array.fill[Int](inputDims + 1)(3),
      Array.fill[Int](inputDims + 1)(5),
      Array.fill[Int](inputDims + 1)(7))
    for (p <- params)
      time { testWMModel(trainingData, testData, p) }

    // Stop spark
    sc.stop()
  }
}
