(ns tech.v3.libs.smile.svm
(:require
   [tech.v3.datatype :as dt]
   [tech.v3.datatype.errors :as errors]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.libs.smile.discrete-nb :as nb]
   [tech.v3.libs.smile.nlp :as nlp]
   [tech.v3.ml :as ml])
(:import [smile.classification SVM]
           [smile.data SparseDataset]
           [smile.util SparseArray])
  )




(defn train [feature-ds target-ds options]
  "Training function of  SVM model. "
  (let [train-data
        (into-array
         (map
          double-array
          (ds/value-reader feature-ds)))]

    (SVM/fit train-data
             (into-array Integer/TYPE (seq  (get target-ds (first (ds-mod/inference-target-column-names target-ds)))))
             ^double (get options :smile.svm.C 1.0)
             ^double (get options :smile.svm.tol 1e-4)
             )))

(defn predict [feature-ds
               thawed-model
               model]
  "Predict function for SVM model"
  (let [to-predict-data
        (into-array
         (map
          double-array
          (ds/value-reader feature-ds)))

        target-colum (first (:target-columns model))
        predictions (.predict (:model-data model) to-predict-data)]
    (ds/->dataset {target-colum predictions} )
    )
  )


(ml/define-model!
  :smile.classification/svm
  train
  predict
  {})


(comment
  (do
    (require '[tech.v3.dataset.column-filters :as cf])
    (require '[tech.v3.dataset.modelling :as ds-mod])
    (require '[tech.v3.ml.loss :as loss])
    (def src-ds (ds/->dataset "test/data/iris.csv"))
    (def ds (->  src-ds
                 (ds/add-or-update-column
                  (ds/new-column "species"
                                 (map
                                  #(if  (= "setosa" %) 1 -1)
                                  (get src-ds "species"))
                                 ))
                 (ds-mod/set-inference-target "species")))

    (def feature-ds (cf/feature ds))
    (def split-data (ds-mod/train-test-split ds))
    (def train-ds (:train-ds split-data))
    (def test-ds (:test-ds split-data))
    (def model (ml/train train-ds {:model-type :smile.classification/svm}))
    (def prediction (ml/predict test-ds model))

    ))
