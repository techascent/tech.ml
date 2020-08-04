(ns tech.libs.smile.protocols
  (:require [tech.libs.smile.data :as smile-data]
            [tech.v2.datatype :as dtype])
  (:import [smile.data.formula Formula]
           [smile.data.type StructType]
           [smile.regression DataFrameRegression]
           [smile.classification DataFrameClassifier]
           [java.util Properties List]
           [smile.data.formula Formula TechFactory Variable]))


(set! *warn-on-reflection* true)


(defprotocol PToFormula
  (get-model-formula [item]))


(extend-protocol PToFormula
  DataFrameRegression
  (get-model-formula [item] (.formula item))
  DataFrameClassifier
  (get-model-formula [item] (.formula item)))


(defn ->formula
  ^Formula [item]
  (if (instance? Formula item)
    item
    (get-model-formula item)))


(defn initialize-model-formula!
  [model options]
  (let [colmap (:column-map options)
        formula (->formula model)
        ^List fields (->> colmap
                          (mapv (fn [[k {:keys [datatype]}]]
                                  (smile-data/smile-struct-field
                                   k datatype))))
        struct-type (StructType. fields)]
    (.bind formula struct-type)))


(defn- resolve-default
  [item dataset]
  (if (fn? item)
    (item dataset)
    item))


(defn options->properties
  ^Properties [metadata dataset options]
  (let [pname-stem (:property-name-stem metadata)]
    (->> (:options metadata)
         (reduce (fn [^Properties props {:keys [name default]}]
                   (.put props (format "%s.%s"
                                       pname-stem
                                       (.replace ^String (clojure.core/name name)
                                                 "-" "."))
                         (str (dtype/cast (or (get options name)
                                              (resolve-default default dataset))
                                          (dtype/get-datatype default))))
                   props)
                 (Properties.)))))


(defn make-formula
  "Make a formula out of a response name and a sequence of feature names"
  [^String response & [features]]
  (Formula. (TechFactory/variable response)
            ^"[Lsmile.data.formula.Variable;" (->> features
                                                   (map #(TechFactory/variable %))
                                                   (into-array Variable ))))
