(defproject techascent/tech.ml "0.5-SNAPSHOT"
  :description "Base concepts of the techascent ml suite"
  :url "http://github.com/techascent/tech.ml-base"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [techascent/tech.compute "3.7"]
                 [techascent/tech.io "2.8"]
                 [camel-snake-kebab "0.4.0"]
                 [org.apache.commons/commons-math3 "3.6.1"]
                 [tech.tablesaw/tablesaw-core "0.30.2"]
                 [ml.dmlc/xgboost4j "0.81"]
                 [com.github.haifengl/smile-core "1.5.2"]]
  :java-source-paths ["java"])
