(ns tech.ml.registry
  (:require [tech.ml.protocols :as proto]))

(def ^:dynamic *registered-systems* (atom {}))


(defn system
  [system-name]
  (let [system-name (if-let [ns-name (namespace system-name)]
                      (keyword ns-name)
                      system-name)]
    (if-let [retval (get @*registered-systems* system-name)]
      retval
      (throw (ex-info (format "Failed to find system.  Perhaps a require is missing?" )
                      {:system-name system-name})))))


(defn register-system
  [system]
  (swap! *registered-systems* assoc (proto/system-name system) system)
  (proto/system-name system))


(defn system-names
  []
  (->> (keys @*registered-systems*)
       set))
