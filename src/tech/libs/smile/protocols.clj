(ns tech.libs.smile.protocols
  (:require [tech.libs.smile.data :as smile-data])
  (:import [smile.data.formula Formula]
           [smile.data.type StructType]))


(defprotocol PToFormula
  (get-model-formula [item]))


(defn ->formula
  ^Formula [item]
  (if (instance? Formula item)
    item
    (get-model-formula item)))


(defn initialize-model-formula!
  [model options]
  (let [colmap (:column-map options)
        formula (->formula model)
        struct-type (StructType.
                     ^"[Lsmile.data.type.StructField;"
                     (->> colmap
                          (map (fn [[k {:keys [datatype]}]]
                                 (smile-data/smile-struct-field
                                  k datatype)))
                          (into-array)))]
    (.bind formula struct-type)))
