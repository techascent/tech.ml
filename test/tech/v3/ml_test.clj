(ns tech.v3.ml-test
  (:require [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.libs.smile.discrete-nb :as nb]
            [tech.v3.libs.smile.nlp :as nlp]
            [tech.v3.ml :as ml]
            [tech.v3.ml.gridsearch :as ml-gs]))

(comment

  (def dataset
    (->
     (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword })
     (ds/select-columns [:Text :Score])
     (ds/update-column :Score #(map dec %))
     (nlp/count-vectorize :Text :bow nlp/default-text->bow)
     (ds-mod/set-inference-target :Score)))


  (defn preprocess [ds options]
    (-> ds
        (nb/bow->SparseArray :bow :bow-sparse (:vocab-size options))))

  (def models
    (ml/train-auto-gridsearch dataset {:model-type :discrete-naive-bayes
                                       :discrete-naive-bayes-model :multinomial
                                       :sparse-column :bow-sparse
                                       :k 5
                                       :preprocess {:fn preprocess
                                                    :vocab-size (ml-gs/linear 100 10000)}

                                       }))

  (ml/train dataset {:model-type :discrete-naive-bayes
                                     :discrete-naive-bayes-model :multinomial
                                     :sparse-column :bow-sparse
                                     :k 5
                                     :preprocess {:fn preprocess
                                                  :vocab-size 1000}

                                     })

  )
