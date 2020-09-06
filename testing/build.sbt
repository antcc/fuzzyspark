name := "fuzzyspark-test"
version := "1.0"
scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.3.2" % "provided",
  "org.apache.spark" %% "spark-mllib" % "2.3.2" % "provided",
  "com.github.antcc" %% "fuzzyspark" % "1.0"
)

assemblyOption in assembly :=
  (assemblyOption in assembly).value.copy(includeScala = false)
