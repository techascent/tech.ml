(ns tech.ml.model
  (:require [tech.io.temp-file :as temp-file]
            [tech.resource :as resource]
            [tech.resource.stack :as stack]
            [tech.io :as io]
            [tech.io.url :as io-url])
  (:import [java.io ByteArrayInputStream ByteArrayOutputStream
            ObjectOutputStream ObjectInputStream
            InputStream OutputStream]))


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


(defn model-file-save->byte-array
  "Given a save fn that takes 1 argument, where to save the file,
  produce a byte array."
  ^bytes [save-fn]
  (resource/stack-resource-context
   (let [temp-file (-> (temp-file/temp-resource-file)
                       io-url/url->file-path)]
     (save-fn temp-file)
     (with-open [^InputStream in-s (io/input-stream (io/file temp-file))
                 out-b (ByteArrayOutputStream.)]
       (io/copy in-s out-b)
       (.toByteArray out-b)))))


(defn byte-array-file-load->model
  "Given the saved bytes and a load fn that takes 1 argument, where to load the file,
  produce a model."
  [byte-model load-fn]
  (let [{retval :return-value
         resource-seq :resource-seq}
        (resource/stack-resource-context
         (let [temp-file (-> (temp-file/temp-resource-file)
                             io-url/url->file-path)]
           (with-open [in-s (ByteArrayInputStream. ^bytes byte-model)
                       ^OutputStream outf (io/output-stream (io/file temp-file))]
             (io/copy in-s outf))
           (stack/return-resource-seq (load-fn temp-file))))]
    ;;If there were resource associated with the model load itelf, register those
    ;;with the larger context.
    (when (seq resource-seq)
      (resource/track #(stack/release-resource-seq resource-seq)))
    retval))
