(ns tech.ml.dataset.etl
  (:require [tech.ml.protocols.dataset :as ds-proto]
            [tech.ml.protocols.column :as col-proto]
            [tech.ml.protocols.etl :as etl-proto]
            [tech.ml.dataset.etl.pipeline-operators :as pipeline-operators]
            [tech.ml.dataset.etl.defaults :as defaults])
  (:import [tech.ml.protocols.etl PETLSingleColumnOperator]))


(defn apply-pipeline
  "Returns a map of:
  :pipeline - sequence of {:operation :context}
  :dataset - new dataset."
  [dataset pipeline {:keys [inference?] :as options}]
  (with-bindings {#'defaults/*etl-datatype* (or :float64
                                                (:datatype options))}
    (let [[dataset options] (if-not inference?
                              [(if-let [target-name (:target options)]
                                 (pipeline-operators/set-attribute dataset target-name :target? true)
                                 dataset)
                               ;;Get datatype of all columns initially and full set of columns.
                               (assoc options
                                      :dataset-column-metadata
                                      (mapv col-proto/metadata (ds-proto/columns dataset)))]
                              ;;No change for inference case
                              [dataset options])]
      (merge {:options options}
             (->> pipeline
                  (reduce (partial pipeline-operators/apply-pipeline-operator options)
                          {:pipeline [] :dataset dataset}))))))
