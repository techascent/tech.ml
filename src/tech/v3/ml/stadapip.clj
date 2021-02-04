(ns tech.v3.ml.stadapip
  (:require [tech.v3.ml :as ml]

          [tech.v3.dataset.modelling :as ds-mod]
            )
  )


(defn model [pipeline-ctx options]

  (case (:mode pipeline-ctx)
    :fit (assoc pipeline-ctx
                :tech.v3.ml/model
                (ml/train(:dataset pipeline-ctx)  options))
    :transform (assoc pipeline-ctx
                      :dataset
                      (ml/predict
                       (:dataset pipeline-ctx)
                       (:tech.v3.ml/model pipeline-ctx)
                       ))))

(defn explain [pipeline-ctx]
  (assoc pipeline-ctx
         :tech.v3.ml/mode-explanation
         (ml/explain
          (:tech.v3.ml/model pipeline-ctx)
          ))

  )

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
      (model ctx {:model-type :smile.classification/random-forest})

      )
    (def fitted
      (pipeline {:mode :fit
                 :dataset train-ds}))

    
    (def prediction
      (pipeline (merge fitted
                       {:mode :transform
                        :dataset test-ds}))

      )


    ))
