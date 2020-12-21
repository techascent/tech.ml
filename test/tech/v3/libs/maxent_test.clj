(ns tech.v3.libs.maxent-test
  (:require  [clojure.test :refer [deftest is] :as t]
             [tech.v3.dataset :as ds]
             [tech.v3.dataset.modelling :as ds-mod]
             [tech.v3.libs.smile.nlp :as nlp]
             [tech.v3.libs.smile.maxent :as maxent]
             [tech.v3.ml :as ml]

             ))

(deftest test-maxent []
  (let [reviews
        (->
         (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword })
         (ds/select-columns [:Text :Score])
         (nlp/count-vectorize :Text :bow nlp/default-text->bow)
         (maxent/bow->sparse-array :bow :bow-sparse 1000)
         (ds-mod/set-inference-target :Score)
         )
        trained-model (ml/train reviews {:model-type :maxent
                                         :sparse-column :bow-sparse
                                         })

        ]
    (is (= 1 (get (first (:bow reviews)) "sweet")  ))
    (is (= [120 240 452] (take 3 (first (:bow-sparse reviews)))))
    (is (= 1001 (count  (first (.coefficients (:model-data trained-model))))))

    )
  )
