(defproject techascent/tech.ml.example "0.1.0-SNAPSHOT"
  :description "Base concepts of the techascent ml suite"
  :url "http://github.com/tech-ascent/tech.ml-base"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.1-beta2"]
                 [techascent/tech.ml "1.0-alpha4"]]

  :profiles {:dev {:dependencies [[org.clojure/tools.logging "0.3.1"]
                                  [ch.qos.logback/logback-classic "1.1.3"]]}})
