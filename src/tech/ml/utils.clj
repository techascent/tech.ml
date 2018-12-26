(ns tech.ml.utils)


(defn nanos->millis
  ^long [^long nanos]
  (-> (/ nanos 1000000.0)
      (Math/round)
      long))


(defmacro time-section
  "Time a section, return
  {:retval retval
  :milliseconds ms}"
  [& body]
  `(let [start-time# (System/nanoTime)
         retval# (do ~@body)
         stop-time# (System/nanoTime)]
     {:retval retval#
      :milliseconds (-> (- stop-time# start-time#)
                        nanos->millis)}))


(defn prefix-merge
  [prefix src-map merge-map]
  (merge src-map
         (->> merge-map
              (map (fn [[item-k item-v]]
                     [(keyword (str prefix "-" (name item-k))) item-v]))
              (into {}))))
