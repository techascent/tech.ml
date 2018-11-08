(ns tech.ml-base
  (:require [tech.ml.registry :as registry]
            [tech.ml.protocols :as protocols]
            [tech.ml.dataset :as dataset])
  (:import [java.util UUID]))


(defn train
  [system-name feature-keys label-keys options dataset]
  (let [ml-system (registry/system system-name)
        options (merge options (protocols/coalesce-options ml-system))
        {:keys [coalesced-dataset scale-map options]}
        (dataset/apply-dataset-options feature-keys label-keys options dataset)
        model (protocols/train ml-system options coalesced-dataset)]
    (merge {:system system-name
            :model model
            :options options
            :feature-keys feature-keys
            :label-keys label-keys
            :id (UUID/randomUUID)})))


(defn predict
  [model dataset]
  (let [ml-system (registry/system (:system model))
        trained-model (:model model)
        {:keys [coalesced-dataset]} (dataset/apply-dataset-options
                                     (:feature-keys model) nil (:options model) dataset)]
    (protocols/predict ml-system trained-model coalesced-dataset)))
