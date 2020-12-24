(ns tech.v3.libs.protocols-test
  (:require  [clojure.test :refer [is deftest]]
             [tech.v3.libs.smile.protocols :as proto])
  (:import [smile.base.cart SplitRule]))


(deftest numeric-option []
  (is (=  (.getProperty
           (proto/options->properties
            {:property-name-stem "prefix"
             :options
             [ {:name :test-numeric
                :type :int64
                :default 12}]}
            nil
            nil) "prefix.test.numeric")
          "12")))

(deftest string-option []
  (is (=  (.getProperty
           (proto/options->properties
            {:property-name-stem "prefix"
             :options
             [ {:name :test-string
                :type :string
                :default "hello"}]}
            nil
            nil) "prefix.test.string")
          "hello")))


(deftest lookup-string-option []
  (is (=  (.getProperty
           (proto/options->properties
            {:property-name-stem "prefix"
             :options
             [ {:name :test-lookup
                :type :string
                :lookup-table {:a "A"
                               :b "B"}
                :default :a}]}
            nil
            nil) "prefix.test.lookup")
          "A")))


(deftest lookup-enum-option []
  (is (=  (.getProperty
           (proto/options->properties
            {:property-name-stem "prefix"
             :options
             [ {:name :test-lookup
                :type :enumeration
                :lookup-table {:gini SplitRule/GINI
                               }
                :default :gini}]}
            nil
            nil) "prefix.test.lookup"

           ) "GINI")))

(deftest lookup-enum-option []
  (is (=  (.getProperty
           (proto/options->properties
            {:property-name-stem "prefix"
             :options
             [ {:name :test-lookup
                :type :enumeration
                :lookup-table {:gini SplitRule/GINI
                               :entropy SplitRule/ENTROPY
                               }
                :default :gini}]}
            nil
            {:test-lookup :entropy }
            ) "prefix.test.lookup"

           ) "ENTROPY")))
