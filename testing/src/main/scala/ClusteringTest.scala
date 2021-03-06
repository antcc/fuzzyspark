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
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

object ClusteringTest {

  /** Configuration parameters. */
  val hdfs = true
  val numPartitions = 20 * 19
  val numPartitionsPerGroup = 19 * 2
  val saveFile = false
  val outFile = "output/out_cluster.txt"
  val seed = 2020

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

  /** Train a Fuzzy C Means model. */
  def testFuzzyCMeans(
    data: RDD[Vector],
    numCenters: Int,
    initMode: String,
    chiuInstance: Option[SubtractiveClustering] = None): FuzzyCMeansModel = {
    val fcmModel = FuzzyCMeans.fit(
      data,
      initMode = initMode,
      c = numCenters,
      numPartitions = numPartitions,
      chiuInstance = chiuInstance,
      epsilon = 1e-4,
      seed = seed
    )

    // Print results
    println("--> INIT MODE: " + initMode)
    println("--> NO. OF CENTERS: " + fcmModel.c)
    println("--> LOSS: " + fcmModel.computeLoss(data))
    println("--> NO. OF ITERATIONS: " + fcmModel.trainingIter)
    if (saveFile) {
      printToFile(outFile, centersToString(fcmModel.clusterCenters))
      println("--> SAVED CENTERS TO FILE " + outFile)
    }

    fcmModel
  }

  /** Test the labelling algorithm based on Chiu's model identification. */
  def testModelIdentificationLabels(
    labeledTrainingData: RDD[(Vector, Double)],
    testData: RDD[(Vector, Double)],
    chiu: SubtractiveClustering,
    raModel: Double) = {
    // Strip class labels for clustering
    val trainingData = labeledTrainingData.keys.cache()
    val centers = chiu.chiuIntermediate(trainingData)

    println("[Chiu] No. of clusters found = " + centers.size)

    val labels = centers.map { c =>
      labeledTrainingData.lookup(c).head
    }

    // Prediction
    val model = ModelIdentification(centers, raModel, Option(labels))
    val labelAndPreds = testData.map { case (x, l) =>
      (l, model.predictLabel(x))
    }

    // Classification error
    val testAcc =
      100.0 * labelAndPreds.filter ( r => r._1 == r._2 ).count.toDouble / testData.count()
    println(f"--> ChiuI Test Acc = $testAcc%1.3f%%")
  }

  /** Test FCM based classification algorithm. */
  def testFCMLabels(
    labeledTrainingData: RDD[(Vector, Double)],
    testData: RDD[(Vector, Double)],
    numCluster: Int,
    alphas: Array[Double],
    m: Double,
    initMode: String,
    chiuInstance: Option[SubtractiveClustering] = None) = {
    val model = FuzzyCMeans.fitClassifier(
      labeledTrainingData,
      initMode = initMode,
      c = numCluster,
      alpha = alphas(0),
      m = m,
      numPartitions = numPartitions,
      chiuInstance = chiuInstance,
      matrixNorm = FuzzyCMeans.MAX,
      maxIter = 100,
      epsilon = 1e-4,
      seed = seed)

    for (i <- 0 until alphas.size) {
      var testAcc = 0.0
      timePartial {
        // Recompute center labels
        if (i > 0) {
          model.labels = Option(model.computeLabels(labeledTrainingData, alphas(i)))
        }

        // Prediction
        val labelAndPreds = testData.map { case (x, l) =>
          (l, model.predict(x))
        }

        // Classification error
        testAcc =
          100.0 * labelAndPreds.filter ( r => r._1 == r._2 ).count.toDouble / testData.count()
      }
      val tag =
        if (initMode == FuzzyCMeans.CHIU_INTERMEDIATE) "[FCM + ChiuI]" else "[FCM + Random]"

      println(s"$tag Alpha = ${alphas(i)}, m = $m")
      println(s"$tag No. of centers = ${model.c}")
      println(s"$tag No. of iterations = ${model.trainingIter}")
      println(f"$tag Test Loss = ${model.computeLoss(testData.keys)}%1.3f")
      println(f"$tag Test Acc = $testAcc%1.3f%%\n")

    }
  }

/** Test KMeans-based classification algorithm. */
  def testKMeansLabels(
    labeledTrainingData: RDD[(Vector, Double)],
    testData: RDD[(Vector, Double)],
    k: Int) = {
    val trainingData = labeledTrainingData.keys.cache()
    val model = KMeans.train(
      trainingData,
      k = k,
      maxIterations = 100)

    // Assign labels to clusters by majority class
    val clusters = model.clusterCenters
    val c = clusters.size
    val labels = Array.ofDim[Double](c)

    val labelsCount = labeledTrainingData.map { case (x, l) =>
      ((l, model.predict(x)), 1)
    }.reduceByKey ( _ + _ )

    for (j <- 0 until c) {
      labels(j) = labelsCount.filter ( _._1._2 == j )
        .max()(Ordering[Int].on ( _._2 ))._1._1
    }

    // Compute predictions
    val labelAndPreds = testData.map { case (x, l) =>
      (l, labels(model.predict(x)))
    }

    // Classification error
    val testAcc =
      100.0 * labelAndPreds.filter ( r => r._1 == r._2 ).count.toDouble / testData.count()
    println(f"--> KMeans (k = $k) Test Acc = $testAcc%1.3f%%")
  }

  /** Test Random Forest classification algorithm. */
  def testRandomForestLabels(
    labeledTrainingData: RDD[(Vector, Double)],
    testData: RDD[(Vector, Double)],
    numClasses: Int,
    maxDepth: Int) = {
    val trainingData = labeledTrainingData.map { case (x, l) =>
      LabeledPoint(l, x)
    }.cache()
    val categoricalFeaturesInfo = Map[Int, Int]()  // All features are continuous.
    val numTrees = 200
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxBins = 100

    // Train RandomForest model
    val model = RandomForest.trainClassifier(
      trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed)

    // Evaluate model on test instances
    val labelAndPreds = testData.map { case (x, l) =>
      (l, model.predict(x))
    }

    // Compute classification error
    val testAcc =
      100.0 * labelAndPreds.filter ( r => r._1 == r._2 ).count.toDouble / testData.count()
    println(f"--> RandomForest (# = 200) Test Acc = $testAcc%1.3f%%")
  }

  /** Test Logistic Regression classification algorithm. */
  def testLogisticLabels(
    labeledTrainingData: RDD[(Vector, Double)],
    testData: RDD[(Vector, Double)],
    numClasses: Int) = {
    val trainingData = labeledTrainingData.map { case (x, l) =>
      LabeledPoint(l, x)
    }.cache()

    // Run training algorithm to build the model
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(numClasses)
      .run(trainingData)

    // Evaluate model on test instances
    val labelAndPreds = testData.map { case (x, l) =>
      (l, model.predict(x))
    }

    // Compute classification error
    val testAcc =
      100.0 * labelAndPreds.filter ( r => r._1 == r._2 ).count.toDouble / testData.count()
    println(f"--> Logistic Test Acc = $testAcc%1.3f%%")
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

  /** Measure execution time of a small block. */
  def timePartial[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    val time = (t1 - t0) / (60.0 * 1e9)
    println(f"Elapsed time: $time%1.2f min")
    result
  }

  /** Clustering examples with fuzzyspark. */
  def main(args: Array[String]) = {
    // Spark environment configuration
    val conf = new SparkConf().setAppName("ClusteringTest")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("hdfs:///tmp")

    // Get input file
    var trainingFile = args.headOption.getOrElse {
      Console.err.println("--> NO TRAINING FILE PROVIDED. ABORTING...")
      sys.exit(1)
    }
    if (hdfs)
      trainingFile = "file://" + trainingFile

    // Load labeled training data into RDD with specified number of partitions
    val trainingInput = sc.textFile(trainingFile, numPartitions)
    var labeledTrainingData = trainingInput.map { line =>
      val x = Vectors.dense(line.split(",").map ( _.toDouble ))
      (Vectors.dense(x.toArray.slice(0, x.size - 1)), x(x.size - 1))
    }.cache()
    var trainingData = labeledTrainingData.keys.cache()

    // Test file
    val testFile = if (hdfs) "file://" + args(1) else args(1)
    val testInput = sc.textFile(testFile, numPartitions)
    var testData = testInput.map { line =>
      val x = Vectors.dense(line.split(",").map ( _.toDouble ))
      (Vectors.dense(x.toArray.slice(0, x.size - 1)), x(x.size - 1))
    }.cache()

    var scaler = new StandardScaler(withMean = true, withStd = true).fit(trainingData)

    // Normalize train and test
    labeledTrainingData = labeledTrainingData.map { case (features, label) =>
      (scaler.transform(features), label)
    }
    trainingData = labeledTrainingData.keys.cache()
    testData = testData.map { case (features, label) =>
      (scaler.transform(features), label)
    }

    // Set parameters for the executions
    val numClasses = 2
    val maxDepth = 10

    val ra = 1.5
    val rb = 3.0
    val raGlobal = 1.5
    val rbGlobal = 3.0
    val lb = 0.15
    val ub = 0.5
    val lbGlobal = 0.45
    val ubGlobal = 0.5
    val raModel = 1.5

    val chiu =
      SubtractiveClustering(
        ra, rb, lb, ub, numPartitions,
        numPartitionsPerGroup,
        raGlobal, rbGlobal, lbGlobal, ubGlobal)

    // Run algorithms

    time { testRandomForestLabels(labeledTrainingData, testData, numClasses, maxDepth) }
    time { testLogisticLabels(labeledTrainingData, testData, numClasses) }

    val ks = Array(250, 500, 1000)
    for (k <- ks) {
      time { testKMeansLabels(labeledTrainingData, testData, k) }
    }

    time {
      testModelIdentificationLabels(labeledTrainingData, testData, chiu, raModel)
    }

    var alphas = Array(0.2, 0.4, 0.6, 0.8)
    var ms = Array(1.25, 1.5, 1.75, 2.0, 2.25, 2.5)

    for (m <- ms) {
      time {
        testFCMLabels(labeledTrainingData, testData, 250, alphas, m,
          FuzzyCMeans.RANDOM, None) }
    }

    for (m <- ms) {
      time {
        testFCMLabels(labeledTrainingData, testData, 500, alphas, m,
          FuzzyCMeans.RANDOM, None) }
    }

    sc.stop()
  }
}
