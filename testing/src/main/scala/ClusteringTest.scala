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

import fuzzyspark.clustering.{FuzzyCMeans, FuzzyCMeansModel,
  SubtractiveClustering, ModelIdentification}

import java.io.File
import java.io.PrintWriter
import scala.io.Source
import scala.math.sqrt

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

object ClusteringTest {

  /** Configuration parameters. */
  val hdfs = true
  val numPartitions = 5
  val numPartitionsPerGroup = 10
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
  def testChiuGlobal(data: RDD[Vector], ra: Double) = {
    val chiu = SubtractiveClustering(numPartitions).setRadius(ra)
    val centers = chiu.chiuGlobal(data)

    // Print results
    println("--> NO. OF CENTERS: " + centers.length)
    if (saveFile) {
      printToFile(outFile, centersToString(centers))
      println("--> SAVED CENTERS TO FILE " + outFile)
    }
    else {
      println("--> CLUSTER CENTERS:\n")
      centers.foreach(println)
      println("\n")
    }

    centers
  }

  /** Test the Subtractive Clustering local algorithm. */
  def testChiuLocal(data: RDD[Vector], ra: Double) = {
    val chiu = SubtractiveClustering(numPartitions).setRadius(ra)
    val centers = data.mapPartitionsWithIndex ( chiu.chiuLocal )
      .map ( _._2 )
      .collect()
      .toArray

    // Print results
    println("--> NO. OF CENTERS: " + centers.length)
    if (saveFile) {
      printToFile(outFile, centersToString(centers))
      println("--> SAVED CENTERS TO FILE " + outFile)
    }
    else {
      println("--> CLUSTER CENTERS:\n")
      centers.foreach(println)
      println("\n")
    }
  }

  /** Test the Subtractive Clustering intermediate algorithm. */
  def testChiuIntermediate(data: RDD[Vector], ra: Double) = {
    val chiu = SubtractiveClustering(ra, 0.15, 0.5, numPartitions, numPartitionsPerGroup)
    val centers = chiu.chiuIntermediate(data)

    // Print results
    println("--> NO. OF CENTERS: " + centers.length)
    if (saveFile) {
      printToFile(outFile, centersToString(centers))
      println("--> SAVED CENTERS TO FILE " + outFile)
    }
    else {
      println("--> CLUSTER CENTERS:\n")
      centers.foreach(println)
      println("\n")
    }
  }

  /** Train a Fuzzy C Means model. */
  def testFuzzyCMeans(labeledData: RDD[(Vector, Double)]): FuzzyCMeansModel = {
    val fcmModel = FuzzyCMeans.train(
      labeledData,
      initMode = FuzzyCMeans.RANDOM,
      c = 3,
      numPartitions = numPartitions,
      chiuInstance =
        Option(SubtractiveClustering(numPartitions))
    )

    // Print results
    println("--> NO. OF CENTERS: " + fcmModel.c)
    println("--> LOSS: " + fcmModel.trainingLoss)
    println("--> NO. OF ITERATIONS: " + fcmModel.trainingIter)
    if (saveFile) {
      printToFile(outFile, centersToString(fcmModel.clusterCenters))
      println("--> SAVED CENTERS TO FILE " + outFile)
    }
    else {
      println("--> CLUSTER CENTERS:\n" +
        fcmModel.clusterCenters.map ( _.toString ).mkString("\n") + "\n")
    }

    fcmModel
  }

  /** Test the Model Identification algorithm. */
  def testModelIdentificationOutput(
    trainingData: RDD[Vector],
    testData: RDD[Vector],
    ra: Double) = {
    val chiu = SubtractiveClustering(numPartitions).setRadius(ra)
    val centers = chiu.chiuGlobal(trainingData)

    val output = ModelIdentification(centers, chiu.getRadius)
      .predictOutputRDD(testData)
      .collect()
      .toArray
    println("--> INPUT, OUTPUT DATA:\n")
    for (o <- output) {
      println(o._1 + "," + o._2)
    }
    println("\n")
  }

  /** Test the labelling algorithm based on Chiu's model identification. */
  def testModelIdentificationLabels(
    labeledTrainingData: RDD[(Vector, Double)],
    testData: RDD[(Vector, Double)],
    ra: Double) = {
    // Strip class labels for clustering
    val trainingData = labeledTrainingData.keys.cache()
    val chiu = SubtractiveClustering(numPartitions).setRadius(ra)
    val centers = chiu.chiuGlobal(trainingData)
    val labels = centers.map { c =>
      labeledTrainingData.lookup(c).head
    }

    // Prediction
    val model = ModelIdentification(centers, chiu.getRadius, Option(labels))
    val labelAndPreds = testData.map { case (x, l) =>
      (l, model.predictLabel(x))
    }

    // Classification error
    val testErr =
      100.0 * labelAndPreds.filter ( r => r._1 != r._2 ).count.toDouble / testData.count()
    println(s"--> Chiu Test Error = $testErr%")
  }

  /** Test FCM based classification algorithm. */
  def testFCMLabels(
    labeledTrainingData: RDD[(Vector, Double)],
    testData: RDD[(Vector, Double)],
    numClasses: Int) = {
    // Strip class labels for clustering
    val trainingData = labeledTrainingData.keys.cache()
    val model = FuzzyCMeans.train(
      labeledTrainingData,
      initMode = FuzzyCMeans.RANDOM,
      c = numClasses,
      numPartitions = numPartitions,
      classification = true)

    // Prediction
    val labelAndPreds = testData.map { case (x, l) =>
      (l, model.predict(x))
    }

    // Classification error
    val testErr =
      100.0 * labelAndPreds.filter ( r => r._1 != r._2 ).count.toDouble / testData.count()
    println(s"--> FCM Test Error = $testErr%")
  }

  /** Test Random Forest classification algorithm. */
  def testRandomForestLabels(
    labeledTrainingData: RDD[(Vector, Double)],
    testData: RDD[(Vector, Double)],
    numClasses: Int) = {
    val trainingData = labeledTrainingData.map { case (x, l) => LabeledPoint(l, x) }.cache()
    val numClasses = 3
    val categoricalFeaturesInfo = Map[Int, Int]()  // All features are continuous.
    val numTrees = 20
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    // Train RandomForest model
    val model = RandomForest.trainClassifier(
      trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances
    val labelAndPreds = testData.map { case (x, l) =>
      (l, model.predict(x))
    }

    // Compute classification error
    val testErr =
      100.0 * labelAndPreds.filter ( r => r._1 != r._2 ).count.toDouble / testData.count()
    println(s"--> RandomForest Test Error = $testErr%")
  }

  /** Test SVM classification algorithm. */
  def testLogisticLabels(
    labeledTrainingData: RDD[(Vector, Double)],
    testData: RDD[(Vector, Double)],
    numClasses: Int) = {
    val trainingData = labeledTrainingData.map { case (x, l) => LabeledPoint(l, x) }.cache()

    // Run training algorithm to build the model
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(numClasses)
      .run(trainingData)

    // Evaluate model on test instances
    val labelAndPreds = testData.map { case (x, l) =>
      (l, model.predict(x))
    }

    // Compute classification error
    val testErr =
      100.0 * labelAndPreds.filter ( r => r._1 != r._2 ).count.toDouble / testData.count()
    println(s"--> Logistic Test Error = $testErr%")
  }

  /** Clustering examples with fuzzyspark. */
  def main(args: Array[String]) = {
    // Spark environment configuration
    val conf = new SparkConf().setAppName("ClusteringTest")
    val sc = new SparkContext(conf)

    // Get input file
    var trainingFile = args.headOption.getOrElse {
      Console.err.println("--> NO TRAINING FILE PROVIDED. ABORTING...")
      sys.exit(1)
    }
    if (hdfs)
      trainingFile = "file://" + trainingFile

    // Load labeled training data into RDD with specified number of partitions
    val trainingInput = sc.textFile(trainingFile, numPartitions)
    val labeledTrainingData = trainingInput.map { line =>
      val x = Vectors.dense(line.split(",").map ( _.toDouble ))
      (Vectors.dense(x.toArray.slice(0, x.size - 1)), x(x.size - 1))
    }.cache()
    val trainingData = labeledTrainingData.keys.cache()

    // Test file
    val testFile = if (hdfs) "file://" + args(1) else args(1)
    val testInput = sc.textFile(testFile, numPartitions)
    val testData = testInput.map { line =>
      val x = Vectors.dense(line.split(",").map ( _.toDouble ))
      (Vectors.dense(x.toArray.slice(0, x.size - 1)), x(x.size - 1))
    }.cache()

    // Test clustering functions
    //testChiuGlobal(trainingData, 1)
    //testFuzzyCMeans(labeledTrainingData)

    // Test classification functions
    testModelIdentificationLabels(labeledTrainingData, testData, 1)
    testFCMLabels(labeledTrainingData, testData, 3)
    testRandomForestLabels(labeledTrainingData, testData, 3)
    testLogisticLabels(labeledTrainingData, testData, 3)

    // Stop spark
    sc.stop()
  }
}
