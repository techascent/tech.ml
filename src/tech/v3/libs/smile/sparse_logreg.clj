(ns tech.v3.libs.smile.sparse-logreg
  (:require
   [tech.v3.datatype :as dt]
   [clojure.test :refer :all]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.libs.smile.discrete-nb :as nb]
   [tech.v3.libs.smile.nlp :as nlp]
   [tech.v3.ml :as ml])

  (:import [smile.classification SparseLogisticRegression]
           [smile.data SparseDataset]
           [smile.util SparseArray]))




(defn train [feature-ds target-ds options]
  "Training function of sparse logistic regression model.
   The column of name `(options :sparse-column)` of `feature-ds` needs to contain the text as SparseArrays
   over the vocabulary."
  (let [train-array (into-array SparseArray
                                (get feature-ds (:sparse-column options)))
        train-dataset (SparseDataset/of (seq train-array) (options :n-sparse-columns))
        score (get target-ds (first (ds-mod/inference-target-column-names target-ds)))]
    (SparseLogisticRegression/fit train-dataset
                                  (dt/->int-array score)
                                  (get options :lambda 0.1)
                                  (get options :tolerance 1e-5)
                                  (get options :max-iterations 500)
                                  )))
(defn predict [feature-ds
               thawed-model
               model]
  "Predict function for sparse logistic regression model."
  (nb/predict feature-ds thawed-model model))


(ml/define-model!
  :smile.classification/sparse-logistic-regression
  train
  predict
  {})


(comment

  (defn get-reviews []
    (->
     (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword })
     (ds/select-columns [:Text :Score])
     (ds/update-column :Score #(map dec %))
     (nlp/count-vectorize :Text :bow nlp/default-text->bow)
     (nb/bow->SparseArray :bow :bow-sparse 100)
     (ds-mod/set-inference-target :Score)))


  (def trained-model
    (ml/train reviews {:model-type :sparse-logistic-regression
                       :sparse-column :bow-sparse}))

  (ml/predict reviews trained-model)

  )
