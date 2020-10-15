(defproject techascent/tech.ml "5.00-alpha1-SNAPSHOT"
  :description "Basic machine learning toolkit"
  :url "http://github.com/techascent/tech.ml-base"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.2-alpha1"]
                 [techascent/tech.ml.dataset "5.00-alpha1"]
                 [ml.dmlc/xgboost4j "0.90"]]
  :profiles
  {:codox
   {:dependencies [[codox-theme-rdash "0.1.2"]]
    :plugins [[lein-codox "0.10.7"]]
    :codox {:project {:name "tech.ml"}
            :metadata {:doc/format :markdown}
            :themes [:rdash]
            :source-paths ["src"]
            :output-path "docs"
            :doc-paths ["topics"]
            :source-uri "https://github.com/techascent/tech.ml/blob/master/{filepath}#L{line}"
            :namespaces [tech.v3.ml]}}}
  :aliases {"codox" ["with-profile" "codox,dev" "codox"]}
  :java-source-paths ["java"])
