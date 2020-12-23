(ns tech.v3.libs.smile.maxent
  (:require [pppmap.core :as ppp]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.libs.smile.nlp :as nlp]
            [tech.v3.ml :as ml])
  (:import smile.classification.Maxent))

(def maxent-default-parameters
  {
   :lambda 0.1
   :tol 1e-5
   :max-iter 500
   })

(defn bow->sparse-indices [bow vocab->index-map]
  "Converts the token-frequencies to the sparse vectors
   needed by Maxent"
  (->>
   (merge-with
    (fn [index count]
      [index count])
    vocab->index-map
    bow)
   vals
   (filter vector?)
   (map first)
   (into-array Integer/TYPE)))


(defn bow->sparse-array [ds bow-col indices-col vocab-size]
  "Converts a bag-of-word column `bow-col` to sparse indices column `indices-col`,
   as needed by the Maxent model.
   `vocab size` is the size of vocabluary used, sorted by token frequency "
  (nlp/bow->something-sparse ds bow-col indices-col vocab-size bow->sparse-indices))


(defn maxent-train [feature-ds target-ds options maxent-type]
    "Training function of Maxent model
   The column of name `(options :sparse-colum)` of `feature-ds` needs to contain the text as a sparce vector
   agains the vocabulary."
  (let [train-array (into-array ^"[[Ljava.lang.Integer"
                                (get feature-ds (:sparse-column options)))
        train-score-array (into-array Integer/TYPE
                                      (get target-ds (first (ds-mod/inference-target-column-names target-ds))))
        p (count (-> feature-ds meta :count-vectorize-vocabulary :vocab->index-map))
        options (merge maxent-default-parameters options)]
    (case maxent-type
      :multinomial
      (Maxent/multinomial
       p
       train-array
       train-score-array
       (:lambda options)
       (:tol options)
       (:max-iter options))
      :binomial
      (Maxent/binomial
       p
       train-array
       train-score-array
       (:lambda options)
       (:tol options)
       (:max-iter options)))))

(defn maxent-train-multinomial [feature-ds target-ds options]
  "Training function of Maxent/multinomial model
   The column of name `(options :sparse-colum)` of `feature-ds` needs to contain the text as a sparse vector
   agains the vocabulary."
  (maxent-train feature-ds target-ds options :multinomial))


(defn maxent-train-binomial [feature-ds target-ds options]
  "Training function of Maxent/binomial model
   The column of name `(options :sparse-colum)` of `feature-ds` needs to contain the text as a sparse vector
   agains the vocabulary."
  (maxent-train feature-ds target-ds options :binomial))


(defn maxent-predict [feature-ds
                      thawed-model
                      model]
  "Predict function for Maxent"
  (let [predict-array
        (into-array ^"[[Ljava.lang.Integer"
                    (get feature-ds :bow-sparse))
        target-colum (first (:target-columns model))]
    (ds/->dataset {
                   target-colum
                   (seq  (.predict (:model-data model) predict-array))})))


(ml/define-model!
  :maxent-multinomial
  maxent-train-multinomial
  maxent-predict
  {})

(ml/define-model!
  :maxent-binomial
  maxent-train-binomial
  maxent-predict
  {})



(comment

  ;; train + predict
  (def reviews
    (->
     (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword })
     (ds/select-columns [:Text :Score])
     (nlp/count-vectorize :Text :bow nlp/default-text->bow)
     (bow->sparse-array :bow :bow-sparse 1000)
     (ds-mod/set-inference-target :Score)
     ))

  (def trained-model (ml/train reviews {:model-type :maxent-multinomial
                                        :sparse-column :bow-sparse
                                        }))
  ;; should predict on new data
  (ml/predict reviews trained-model))


(comment
  ;;  grid search
  (def models
    (ml/train-auto-gridsearch
     reviews
     {:model-type :maxent-multinomial
      :sparse-column :bow-sparse
      :lambda {:tech.v3.ml.gridsearch/type :linear
               :start 0.001
               :end 100.0
               :n-steps 30
               :result-space :float64
               }
      :tol  {:tech.v3.ml.gridsearch/type :linear
             :start 1.0E-9
             :end 0.1
             :n-steps 20
             :result-space :float64
             }
      :max-iter {:tech.v3.ml.gridsearch/type :linear
                 :start 100.0
                 :end 10000.0
                 :n-steps 20
                 :result-space :int64 }
      })))
