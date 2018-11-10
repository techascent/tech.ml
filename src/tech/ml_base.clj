(ns tech.ml-base
  (:require [tech.ml.registry :as registry]
            [tech.ml.protocols :as protocols]
            [tech.ml.dataset :as dataset])
  (:import [java.util UUID]))


(defn options->label-map
  [options label-keys]
  (let [label-keys (dataset/normalize-keys label-keys)]
    (when-not (= 1 (count label-keys))
      (throw (ex-info "More than 1 label detected"
                      {:label-keys label-keys})))
    (if-let [retval (get-in options [:label-map (first label-keys)])]
      retval
      (throw (ex-info "Failed to find label map"
                      {:label-keys label-keys
                       :label-maps (get options :label-map)})))))


(defn train
  [system-name feature-keys label-keys options dataset]
  (let [ml-system (registry/system system-name)
        options (merge options (protocols/coalesce-options ml-system))
        {:keys [coalesced-dataset options]}
        (dataset/apply-dataset-options feature-keys label-keys options dataset)
        model (protocols/train ml-system options label-keys coalesced-dataset)]
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
    (protocols/predict ml-system
                       (:options model)
                       (:label-keys model)
                       (:model model)
                       coalesced-dataset)))
