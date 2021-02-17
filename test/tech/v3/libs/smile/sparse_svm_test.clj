(ns tech.v3.libs.smile.sparse-svm-test
  (:require [clojure.test :refer :all]
            [tech.v3.dataset :as ds]

            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.libs.smile.discrete-nb :as nb]
            [tech.v3.libs.smile.nlp :as nlp]
            [tech.v3.ml :as ml]
            [tech.v3.libs.smile.sparse-svm]))

(defn get-reviews []
    (->
     (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword})
     (ds/select-columns [:Text :Score])
     (ds/update-column :Score #(map
                                (fn [val] (if (> val 1)
                                            +1 -1))
                                %))
     (nlp/count-vectorize :Text :bow nlp/default-text->bow)
     (nb/bow->SparseArray :bow :bow-sparse #(nlp/->vocabulary-top-n % 100))
     (ds-mod/set-inference-target :Score)))

(deftest does-not-crash
  (let [reviews (get-reviews)
        trained-model
        (ml/train reviews {:model-type :smile.classification/sparse-svm
                           :sparse-column :bow-sparse
                           :p 100 ;; (-> reviews meta :count-vectorize-vocabulary :index->vocab-map count)
                           })
        freqs (frequencies (:Score (ml/predict reviews trained-model)))
        ]
    (is (pos? (get freqs -1)))
    (is (pos? (get freqs 1)))))
