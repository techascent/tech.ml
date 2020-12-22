(ns tech.v3.libs.discrete-nb-test
  (:require [clojure.test :refer :all]

            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.libs.smile.discrete-nb :as nb]
            [tech.v3.libs.smile.nlp :as nlp]
            [tech.v3.ml :as ml])
  (:import [smile.util SparseArray SparseArray$Entry ]))

(defn get-reviews []
  (->
   (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword })
   (ds/select-columns [:Text :Score])
   (ds/update-column :Score #(map dec %))
   (nlp/count-vectorize :Text :bow nlp/default-text->bow)))




(deftest freqs->SparseArray []
  (is (=
       (map
        #(vector (.i %) (.x %))
        (iterator-seq
         (.iterator (nb/freqs->SparseArray {"a" 10.0  "b" 70.0 "c" 50.0} {"a" 0 "b" 1} ))))
       [[0 10.0] [1 70.0] ])))


(deftest test-discrete-nb
  (is (=
       (:Score
        (let [reviews (->  (get-reviews)
                           (nb/bow->SparseArray :bow :bow-sparse 100)
                           (ds-mod/set-inference-target :Score))
              trained-model
              (ml/train reviews {:model-type :discrete-naive-bayes
                                 :sparse-column :bow-sparse
                                 })
              prediction (ml/predict (ds/head reviews 10) trained-model)
              ]
          prediction
          ))
       [3 3 3 3 3 4 4 3 3 3])))
