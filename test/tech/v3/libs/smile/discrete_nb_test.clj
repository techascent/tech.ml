(ns tech.v3.libs.smile.discrete-nb-test
  (:require
   [clojure.test :refer :all]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.libs.smile.discrete-nb :as nb]
   [tech.v3.libs.smile.nlp :as nlp]
   [tech.v3.ml.gridsearch :as ml-gs]
   [tech.v3.ml :as ml]))


(defn get-reviews []
  (->
   (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword })
   (ds/select-columns [:Text :Score])
   (ds/update-column :Score #(map dec %))
   (nlp/count-vectorize :Text :bow nlp/default-text->bow)
   (nb/bow->SparseArray :bow :bow-sparse #(nlp/->vocabulary-top-n % 100) )
   (ds-mod/set-inference-target :Score)))





(deftest test-discrete-nb-bernoulli
  (is (=
       (:Score
        (let [reviews (get-reviews)
              trained-model
              (ml/train reviews {:model-type :discrete-naive-bayes
                                 :discrete-naive-bayes-model :bernoulli
                                 :sparse-column :bow-sparse
                                 :p 100
                                 :k 5})
              prediction (ml/predict (ds/head reviews 10) trained-model)]
          prediction))
       [3 3 3 3 3 4 4 3 3 3])))

(deftest test-discrete-nb-multinomial
  (is (=
       (:Score
        (let [reviews (get-reviews)
              trained-model
              (ml/train reviews {:model-type :discrete-naive-bayes
                                 :discrete-naive-bayes-model :multinomial
                                 :sparse-column :bow-sparse
                                 :p 100
                                 :k 5})
              prediction (ml/predict (ds/head reviews 10) trained-model)]
          prediction))
       [4 4 3 2 3 4 4 4 4 4])))
