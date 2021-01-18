(defproject techascent/tech.ml "5.00-beta-13-SNAPSHOT"
  :description "Basic machine learning toolkit.  `tech.v3.ml` is the root
namespace and provides train/predict pathways based on datasets and
an options map.  Please see the xgboost article for a quick runthough
of how to use this library."
  :url "http://github.com/techascent/tech.ml-base"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.2-alpha1"]
                 [techascent/tech.ml.dataset "5.00-beta-25"]
                 [ml.dmlc/xgboost4j_2.12 "1.3.1"]
                 [ml.dmlc/xgboost4j-spark_2.12 "1.3.1"]
                 [org.bytedeco/openblas "0.3.10-1.5.4"]
                 [org.apache.hadoop/hadoop-common "3.3.0" :exclusions [org.apache.commons/commons-math3]]
                 [org.bytedeco/openblas-platform "0.3.10-1.5.4"]
                 [pppmap/pppmap "0.2.1"]]


  :profiles
  {:codox
   {:dependencies [[codox-theme-rdash "0.1.2"]]
    :plugins [[lein-codox "0.10.7"]]
    :codox {:project {:name "tech.ml"}
            :metadata {:doc/format :markdown}
            :namespaces [tech.v3.ml
                         tech.v3.ml.metrics
                         tech.v3.ml.loss
                         tech.v3.ml.gridsearch
                         tech.v3.libs.xgboost
                         tech.v3.libs.smile.classification
                         tech.v3.libs.smile.regression]
            :themes [:rdash]
            :source-paths ["src"]
            :output-path "docs"
            :doc-paths ["topics"]
            :source-uri "https://github.com/techascent/tech.ml/blob/master/{filepath}#L{line}"}}}
  :aliases {"codox" ["with-profile" "codox,dev" "codox"]}
  :java-source-paths ["java"])
