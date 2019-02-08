(ns tech.ml.protocols.column)

(defprotocol PIsColumn
  (is-column? [item]))


(extend-protocol PIsColumn
  Object
  (is-column? [item] false))


(defprotocol PColumn
  (column-name [col])
  (supported-stats [col]
    "List of available stats for the column")
  (metadata [col]
    "Return the metadata map for this column.
    Metadata must contain :name :type :size.  Categorical
columns must have :categorical? true and the inference target
should have :target? true.")
  (set-metadata [col data-map]
    "Set the metadata on the column returning a new column.")
  (missing [col]
    "Indexes of missing values")
  (unique [col]
    "Set of all unique values")
  (stats [col stats-set]
    "Return a map of stats.  Stats set is a set of the desired stats in keyword
form.  Guaranteed support across implementations for :mean :variance :median :skew.
Implementations should check their metadata before doing calculations.")
  (column-values [col]
    "Return a 'thing convertible to a sequence' of values for this column.
May be a java array or something else.")
  (set-values [col idx-val-seq]
    "Set values in the column returning a new column with same name and datatype.  Values
which cannot be simply coerced to the datatype are an error.")
  (select [col idx-seq]
    "Return a new column with the subset of indexes")
  (empty-column [col datatype elem-count column-name]
    "Return a new column of this supertype where all values are missing.")
  (new-column [col datatype elem-count-or-values column-name]
    "Return a new column of this supertype with these values")
  (math-context [col]))


(defprotocol PColumnMathContext
  (unary-op [ctx op-arg op-kwd])
  (binary-op [ctx op-args op-scalar-fn op-kwd]))
