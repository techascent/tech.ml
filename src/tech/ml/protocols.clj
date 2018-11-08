(ns tech.ml.protocols)


(defprotocol PMLSystem
  (system-name [system])
  (coalesce-options [system]
    "Force double or float arrays by setting coalesce options.
See dataset/coalesce-dataset")
  (train [system options-and-hyperparameters coalesced-dataset]
    "dataset->model transformation.  Models are maps with data.")
  (predict [system model coalesced-dataset]))
