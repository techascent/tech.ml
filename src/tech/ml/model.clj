(ns tech.ml.model
  (:import [java.io ByteArrayInputStream ByteArrayOutputStream
            ObjectOutputStream ObjectInputStream]))


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
