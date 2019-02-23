(defproject techascent/tech.ml "0.17-SNAPSHOT"
  :description "Base concepts of the techascent ml suite"
  :url "http://github.com/techascent/tech.ml-base"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [techascent/tech.ml.dataset "0.15"]
                 [techascent/tech.io "2.8"]
                 [ml.dmlc/xgboost4j "0.81"]]

  :profiles {:dev {:dependencies [[org.clojure/tools.logging "0.3.1"]
                                  [ch.qos.logback/logback-classic "1.1.3"]]}}

  :java-source-paths ["java"])
