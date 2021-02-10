(ns tech.v3.ml.metamorph
  (:require [tech.v3.ml :as ml]))


(defn model [ options]
  (fn [pipeline-ctx]
    (let [id (:metamorph/id pipeline-ctx)]
      (case (:metamorph/mode pipeline-ctx)
        :metamorph/fit (assoc pipeline-ctx
                    id
                    (ml/train (:metamorph/data pipeline-ctx)  options))
        :metamorph/transform (assoc pipeline-ctx
                          :metamorph/data
                          (ml/predict
                           (:metamorph/data pipeline-ctx)
                           (get pipeline-ctx (:metamorph/id pipeline-ctx))))))) )



(comment
  (do
    (require '[tech.v3.dataset.column-filters :as cf])
    (require '[tech.v3.dataset.modelling :as ds-mod])
    (require '[tech.v3.dataset :as ds])
    (require '[tech.v3.ml.loss :as loss])
    (require '[tech.v3.libs.smile.classification])
    
    (def src-ds (ds/->dataset "test/data/iris.csv"))
    (def ds (->  src-ds
                 (ds/categorical->number cf/categorical)
                 (ds-mod/set-inference-target "species")))
    (def feature-ds (cf/feature ds))
    (def split-data (ds-mod/train-test-split ds))
    (def train-ds (:train-ds split-data))
    (def test-ds
      (->
       (:test-ds split-data)
       (ds/add-or-update-column :species 0)
       )

      )


    (defn pipeline [ctx]
      ((model {:model-type :smile.classification/random-forest})
       ctx)

      )
    (def fitted
      (pipeline
       {:metamorph/id "1"
        :metamorph/mode :metamorph/fit
        :metamorph/data train-ds}))

    
    (def prediction
      (pipeline (merge fitted
                       {:metamorph/mode :metamorph/transform
                        :metamorph/data test-ds}))

      )))
