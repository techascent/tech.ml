(ns tech.v3.ml.model
  "Internal namespace of helper functions used to implement models."
  (:require [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [tech.v3.dataset.tensor :as ds-tens]
            [tech.v3.dataset :as ds]
            [clojure.set :as set])
  (:import [java.io ByteArrayInputStream ByteArrayOutputStream
            ObjectOutputStream ObjectInputStream
            InputStream OutputStream]))


(defn options->model-type
  [{:keys [model-type]}]
  (-> model-type
      name
      keyword))


;;This uses java serialization which is fragile between jar versions For Smile, you have
;;no choice for for a lot of toolkits you have the option to use the toolkit's save to
;;file methods which tend to be more robust.
(defn model->byte-array
  ^bytes [model]
  (with-open [result (ByteArrayOutputStream.)
              obj-stream (ObjectOutputStream. result)]
    (.writeObject obj-stream model)
    (.flush obj-stream)
    (.toByteArray result)))


(defn byte-array->model
  [^bytes data]
  (let [in-stream (ByteArrayInputStream. data)
        obj-in-stream (ObjectInputStream. in-stream)]
    (.readObject obj-in-stream)))


(defn finalize-regression
  [reg-tens target-cname]
  (let [n-rows (dtype/ecount reg-tens)]
    (-> (dtt/reshape reg-tens [n-rows 1])
        (ds-tens/tensor->dataset)
        (ds/rename-columns {0 target-cname})
        (ds/update-columnwise :all vary-meta assoc :column-type :prediction)
        (vary-meta assoc :model-type :regression))))


(defn finalize-classification
  [cls-tens n-rows target-cname target-categorical-maps]
  (let [rename-map (-> (get-in target-categorical-maps
                               [target-cname :lookup-table])
                       (set/map-invert))
        n-cols (count rename-map)]
    (-> (dtt/reshape cls-tens [n-rows n-cols])
        (ds-tens/tensor->dataset)
        (ds/rename-columns rename-map)
        (ds/update-columnwise :all vary-meta assoc
                              :column-type :probability-distribution)
        (vary-meta assoc :model-type :classification))))
