(ns tech.v3.libs.smile.discrete-nb

  (:require [pppmap.core :as ppp]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.libs.smile.nlp :as nlp]
            [tech.v3.ml :as ml]
            [tech.v3.libs.smile.maxent :as maxent]
            )
  (:import [smile.classification DiscreteNaiveBayes DiscreteNaiveBayes$Model]
           [smile.util SparseArray])

  )

(defn freqs->SparseArray [freq-map vocab->index-map]
  (let [sparse-array (SparseArray.)]
    (run!
     (fn [[token freq]]
       (when (contains? vocab->index-map token)
         (.append sparse-array ^int (get vocab->index-map token) ^double freq)))
     freq-map)
    sparse-array
    ))

(defn bow->SparseArray [ds bow-col indices-col vocab-size]
  "Converts a bag-of-word column `bow-col` to sparse indices column `indices-col`,
   as needed by the Maxent model."
  (let [vocabulary (nlp/->vocabulary-top-n ds bow-col vocab-size)
        vocab->index-map (:vocab->index-map vocabulary)
        ds
        (vary-meta ds assoc
                   :count-vectorize-vocabulary vocabulary)]
    (ds/add-or-update-column
     ds
     (ds/new-column
      indices-col
      (ppp/ppmap-with-progress
       "bow->SparseArray"
       1000
       #(freqs->SparseArray % vocab->index-map)
       (get ds bow-col))))))


(defn train [feature-ds target-ds options]
  "Training function of Maxent model
   The first feature column of `feature-ds` needs to contain the text as a sparce vector
   agains the vocabulary."
  (let [train-array (into-array SparseArray
                                (get feature-ds (:sparse-column options)))
        train-score-array (into-array Integer/TYPE
                                      (get target-ds (first (ds-mod/inference-target-column-names target-ds))))
        p (count (-> feature-ds meta :count-vectorize-vocabulary :vocab->index-map))
        ;; options (merge maxent-default-parameters options)
        nb (DiscreteNaiveBayes. DiscreteNaiveBayes$Model/BERNOULLI 5 p)]
    (.update nb
             train-array
             train-score-array

             ;; (:lambda options)
             ;; (:tol options)
             ;; (:max-iter options)
             )
    nb
    ))

(defn predict [feature-ds
                      thawed-model
                      model]
  "Predict function for discrete naive bayses"
  (let [predict-array
        (into-array ^"[[Ljava.lang.Integer"
                    (get feature-ds :bow-sparse))
        target-colum (first (:target-columns model))]
    (ds/->dataset (hash-map
                   target-colum
                   (seq  (.predict (:model-data model) predict-array))))))


(ml/define-model!
  :discrete-naive-bayes
  train
  predict
  {})
