(ns tech.v3.ml-test
  (:require [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.libs.smile.discrete-nb :as nb]
            [tech.v3.libs.smile.nlp :as nlp]
            [tech.v3.ml :as ml]
            [tech.v3.ml.gridsearch :as ml-gs]
            [clojure.test :refer (deftest is)]))

(defn preprocess [ds options]
    {:dataset
     (-> ds
         (nlp/count-vectorize :Text :bow nlp/default-text->bow options)
         (nb/bow->SparseArray :bow :bow-sparse #(nlp/->vocabulary-top-n %  (:vocab-size options)))
         )
     :options (merge  options {:a 1})})

(defn get-dataset []
  (->
   (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword })
   (ds/select-columns [:Text :Score])
   (ds/update-column :Score #(map dec %))
   (ds-mod/set-inference-target :Score)))

(deftest grid-search-takes-pre-process-options []

  (let [dataset (get-dataset)
        models
        (ml/train-auto-gridsearch dataset {:model-type :discrete-naive-bayes
                                           :discrete-naive-bayes-model :multinomial
                                           :sparse-column :bow-sparse
                                           :k 5
                                           :preprocess-fn 'tech.v3.ml-test/preprocess
                                           :stopwords (ml-gs/categorical [nil nil :default :google :comprehensive])

                                           :vocab-size (ml-gs/linear 100 10000)}
                                  {:n-gridsearch 5}
                                  )]
    (is (>
         (get-in (first models) [:options :vocab-size]))
        100)))


(deftest  train-takes-pre-process-options
  (let [dataset (get-dataset)
        model
        (ml/train dataset {:model-type :discrete-naive-bayes
                           :discrete-naive-bayes-model :multinomial
                           :sparse-column :bow-sparse
                           :k 5
                           :preprocess-fn 'tech.v3.ml-test/preprocess
                           :vocab-size 1000
                           })]
    (is (=  1 (get-in model [:options :a])))))
