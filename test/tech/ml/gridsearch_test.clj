(ns tech.ml.gridsearch-test
  (:require [tech.ml.gridsearch :as ml-gs]
            [clojure.test :refer :all]))

(deftest gridsearch
  (is (= [{:a {:c {:d 5}, :b 1}, :e :ww}
          {:a {:c {:d 5}, :b 100}, :e :yy}
          {:a {:c {:d 5}, :b 1000}, :e :xx}
          {:a {:c {:d 5}, :b 10}, :e :zz}
          {:a {:c {:d 5}, :b 31}, :e :xx}]
         (->> (ml-gs/gridsearch {:a {:b (comp long #(* 100 %)
                                              (ml-gs/exp [0.01 100]))
                                     :c {:d 5}}
                                 :e (ml-gs/nominative [:ww :xx :yy :zz])})
              (take 5)
              vec)))

  ;;You can start at any index you want to continue the search.
  (is (= [{:a {:c {:d 5}, :b 10}, :e :zz}
          {:a {:c {:d 5}, :b 31}, :e :xx}]
         (->> (ml-gs/gridsearch {:a {:b (comp long #(* 100 %)
                                              (ml-gs/exp [0.01 100]))
                                     :c {:d 5}}
                                 :e (ml-gs/nominative [:ww :xx :yy :zz])}
                                2)
              (take 2)
              vec))))
