(ns tech.v3.libs.smile.sparse-logreg-test
  (:require [clojure.test :refer :all]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.libs.smile.discrete-nb :as nb]
            [tech.v3.libs.smile.nlp :as nlp]
            [tech.v3.ml :as ml]
            [tech.v3.libs.smile.sparse-logreg]))

(defn get-reviews []
  (->
   (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword })
   (ds/select-columns [:Text :Score])
   (ds/update-column :Score #(map dec %))
   (nlp/count-vectorize :Text :bow nlp/default-text->bow)
   (nb/bow->SparseArray :bow :sparse #(nlp/->vocabulary-top-n % 100))
   (ds-mod/set-inference-target :Score)))



(deftest does-not-crash
  (let [reviews (get-reviews)
        trained-model
        (ml/train reviews {:model-type :smile.classification/sparse-logistic-regression
                           :n-sparse-columns 100
                           :sparse-column :sparse})]

    (is (= [4 4 4 2]
           (take 4
                 (:Score  (ml/predict reviews trained-model)))))))
