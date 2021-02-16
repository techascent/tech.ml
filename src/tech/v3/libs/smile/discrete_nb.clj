(ns tech.v3.libs.smile.discrete-nb
  (:require [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.libs.smile.nlp :as nlp]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.ml :as ml])
  (:import [smile.classification DiscreteNaiveBayes DiscreteNaiveBayes$Model]
           smile.util.SparseArray))



(defn bow->SparseArray [ds bow-col indices-col create-vocab-fn]
  "Converts a bag-of-word column `bow-col` to sparse indices column `indices-col`,
   as needed by the discrete naive bayes model. `vocab size` is the size of vocabluary used, sorted by token frequency "
  (nlp/bow->something-sparse ds bow-col indices-col create-vocab-fn nlp/freqs->SparseArray))






(defn train [feature-ds target-ds options]
  "Training function of discrete naive bayes model.
   The column of name `(options :sparse-colum)` of `feature-ds` needs to contain the text as SparseArrays
   over the vocabulary."
  (let [train-array (into-array SparseArray
                                (get feature-ds (:sparse-column options)))
        train-score-array (into-array Integer/TYPE
                                      (get target-ds (first (ds-mod/inference-target-column-names target-ds))))
        p (:p options)
        _ (errors/when-not-error (pos? p) "p needs to be specified in options and greater 0")
        nb-model
        (case (options :discrete-naive-bayes-model)
          :polyaurn DiscreteNaiveBayes$Model/POLYAURN
          :wcnb DiscreteNaiveBayes$Model/WCNB
          :cnb DiscreteNaiveBayes$Model/CNB
          :twcnb DiscreteNaiveBayes$Model/TWCNB
          :bernoulli  DiscreteNaiveBayes$Model/BERNOULLI
          :multinomial DiscreteNaiveBayes$Model/MULTINOMIAL)
        nb (DiscreteNaiveBayes. nb-model (:k options) p)]
    (.update nb
             train-array
             train-score-array)
    nb))

(defn predict [feature-ds
                      thawed-model
                      model]
  "Predict function for discrete naive bayes"
  (let [sparse-arrays (get feature-ds  (get-in model [:options :sparse-column]))
        target-colum (first (:target-columns model))
        predictions (map #(.predict thawed-model %) sparse-arrays)
        ]
    (ds/->dataset {target-colum predictions})) )


(ml/define-model!
  :discrete-naive-bayes
  train
  predict
  {})
