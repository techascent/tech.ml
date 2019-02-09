(ns tech.ml.details
  (:require [tech.ml.dataset :as dataset]
            [tech.ml.protocols.dataset :as ds-proto]
            [tech.ml.protocols.column :as col-proto]))


(defn options->label-map
  [options label-keys]
  (let [label-keys (dataset/normalize-keys label-keys)]
    (when-not (= 1 (count label-keys))
      (throw (ex-info (format "More than 1 label detected: %s" label-keys)
                      {:label-keys label-keys
                       :options options})))
    (if-let [retval (get-in options [:label-map (first label-keys)])]
      retval
      (throw (ex-info "Failed to find label map"
                      {:label-keys label-keys
                       :label-maps (get options :label-map)})))))


(defn get-target-label-map
  [options]
  (let [label-keys (:label-keys options)
        _ (when-not (= 1 (count label-keys))
            (throw (ex-info "Missing label keys" {})))]
    (if-let [retval (get-in options [:column-metadata (first label-keys) :label-map])]
      retval
      (throw (ex-info (format "Failed to find label map for column %s"
                              (first label-keys))
                      {})))))
