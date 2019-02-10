(ns tech.ml.dataset.etl
  (:require [tech.ml.protocols.dataset :as ds-proto]
            [tech.ml.protocols.column :as col-proto]
            [tech.ml.protocols.etl :as etl-proto]
            [tech.ml.dataset.etl.pipeline-operators :as pipeline-operators]
            [tech.ml.dataset.etl.column-filters :as column-filters]
            [tech.ml.dataset.etl.defaults :as defaults]
            [clojure.set :as c-set])
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
                              [dataset options])
          {:keys [options dataset] :as retval}
          (->> pipeline
               (reduce pipeline-operators/apply-pipeline-operator
                       {:pipeline [] :options options :dataset dataset}))
          target-columns (set (column-filters/execute-column-filter dataset :target?))
          feature-columns (c-set/difference (set (map col-proto/column-name (ds-proto/columns dataset)))
                                            (set target-columns))]

      (assoc retval :options
             (assoc options
                    :feature-columns feature-columns
                    :label-columns target-columns)))))
