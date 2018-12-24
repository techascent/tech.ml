(ns tech.ml.dataset.csv
  (:require [clojure.data.csv :as csv]
            [tech.io :as io]))


(defn csv->dataset
  [file-url & {:keys [nil-val]
               :or {nil-val -1}}]
  (with-open [in-stream (io/reader (io/input-stream file-url))]
    (let [csv-data (csv/read-csv in-stream)
          map-keys (map keyword (first csv-data))
          csv-data (rest csv-data)]
      (->> csv-data
           (mapv (fn [csv-line]
                   (when-not (= (count map-keys)
                                (count csv-line))
                     (throw (ex-info "Line contains bad data" {})))
                   (->> csv-line
                        (map #(try
                                (if (> (count %) 0)
                                  (Double/parseDouble %)
                                  nil-val)
                                (catch Throwable e
                                  (keyword %))))
                        (map vector map-keys)
                        (into {}))))))))
