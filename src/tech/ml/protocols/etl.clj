(ns tech.ml.protocols.etl
  (:require [tech.ml.protocols.column :as col-proto]))


(defprotocol PETLSingleColumnOperator
  "Define an operator for an ETL operation."
  (build-etl-context [op dataset column op-args])
  (perform-etl [op dataset column op-args context]))


(defprotocol PETLMultipleColumnOperator
  (build-etl-context-columns [op dataset column-seq op-args])
  (perform-etl-columns [op dataset column-seq op-args context]))


(defn default-etl-context-columns
  "Default implementation of build-etl-context-columns"
  [op dataset column-seq op-args]
  (->> column-seq
       (map (fn [col]
              [(col-proto/column-name col)
               (build-etl-context op dataset col op-args)]))
       (into {})))


(defn default-perform-etl-columns
  [op dataset column-seq op-args context]
  (->> column-seq
       (reduce (fn [dataset col]
                 (perform-etl op dataset col op-args
                              (get context (col-proto/column-name col))))
               dataset)))


(extend-protocol PETLMultipleColumnOperator
  Object
  (build-etl-context-columns [op dataset column-seq op-args]
    (default-etl-context-columns op dataset column-seq op-args))

  (perform-etl-columns [op dataset column-seq op-args context]
    (default-perform-etl-columns op dataset column-seq op-args context)))
